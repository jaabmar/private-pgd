import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import lsmr
from torch import optim
from torch.optim.lr_scheduler import StepLR

from inference.embedding import Embedding
from inference.privpgd.particle_model import ParticleModel

if TYPE_CHECKING:
    from inference.domain import Domain


class AdvancedSlicedInference:
    def __init__(
        self,
        domain: "Domain",
        hp: Dict[str, Any] = {},
        embedding: Optional["Embedding"] = None,
        constraint_regularizer: Optional[Any] = None,
    ):
        """
        Initialize Advanced Sliced Inference.

        Args:
            domain (Domain): Domain of the dataset.
            N (int): Number of records in the dataset.
            hp (Dict[str, Any]): Hyperparameters.
            embedding (Optional[Embedding], optional): Embedding object. Defaults to None.
            constraint_regularizer (Optional[Any], optional): Constraint regularizer. Defaults to None.
        """
        self.domain = domain
        self.embedding = embedding if embedding else Embedding(domain)
        self.constraint_regularizer = constraint_regularizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_particles = hp["n_particles"] if "n_particles" in hp else 100000
        self.data_init = hp["data_init"] if "data_init" in hp else None
        self.iters = hp["iters"] if "iters" in hp else 1000
        self.lr = hp["lr"] if "lr" in hp else 0.1
        self.batch_size = hp["batch_size"] if "batch_size" in hp else 5
        self.p_mask = hp["p_mask"] if "p_mask" in hp else 80
        self.scale_reg = hp["scale_reg"] if "scale_reg" in hp else 0.0
        self.num_projections = (
            hp["num_projections"] if "num_projections" in hp else 10
        )
        self.scheduler_step = (
            hp["scheduler_step"] if "scheduler_step" in hp else 50
        )
        self.scheduler_gamma = (
            hp["scheduler_gamma"] if "scheduler_gamma" in hp else 0.75
        )
        self.eval_particles = (
            hp["eval_particles"] if "eval_particles" in hp else False
        )
        self.iters_proj = hp["iters_proj"] if "iters_proj" in hp else 1750
        self.num_projections_proj = (
            hp["num_projections_proj"] if "num_projections_proj" in hp else 200
        )
        self.scheduler_step_proj = (
            hp["scheduler_step_proj"] if "scheduler_step_proj" in hp else 100
        )
        self.scheduler_gamma_proj = (
            hp["scheduler_gamma_proj"] if "scheduler_gamma_proj" in hp else 0.8
        )
        self.yprobs = {}
        self.model = None
        self.measurements = []
        self.cliques = []
        self.centers = {}
        self.center_mappings = {}
        self.adjust_for_compression = {}
        self.dimsX = {}
        self.processed_measurements = []
        self.quantized_measurements = []
        self.total = None
        self.history = []
        self.history_particles = []
        self.measurements_processed = []


    def estimate(
        self,
        measurements: List[
            Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]
        ],
        total: Optional[int] = None,
    ) -> Tuple["ParticleModel", float]:
        """
        Estimate the particle model.

        Args:
            measurements (List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]): Measurements for estimation.
            total (Optional[int]): Total number of records, if known. For compatibility
        Returns:
            Tuple["ParticleModel", float]: The estimated particle model and the loss.
        """
        measurements = self.fix_measurements(measurements)
        self._setup(measurements, total)
        loss = self.particle_gradient_descent()

        return self.model, loss

    def fix_measurements(
        self,
        measurements: List[
            Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]
        ],
    ) -> List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]:
        """
        Adjusts and standardizes the format of the measurements.

        Args:
            measurements (List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]): The raw measurements.

        Returns:
            List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]: The standardized measurements.
        """
        assert type(measurements) is list, (
            "Measurements must be a list, given " + measurements
        )
        assert all(
            len(m) == 4 for m in measurements
        ), "Each measurement must be a 4-tuple (Q, y, noise,proj)"

        ans = []
        for Q, y, noise, proj in measurements:
            if type(proj) is list:
                proj = tuple(proj)
            if type(proj) is not tuple:
                proj = (proj,)
            if Q is None:
                Q = sparse.eye(self.domain.size(proj))
            assert np.isscalar(
                noise
            ), "Noise must be a real value, given " + str(noise)
            assert all(a in self.domain for a in proj), (
                str(proj) + " not contained in domain"
            )
            ans.append((Q, y, noise, proj))
        return ans

    def _setup(
        self,
        measurements: List[
            Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]
        ],
        total: Optional[int] = None,
    ):
        """
        Sets up the estimation process by initializing various components based on the given measurements.
        Includes projection step from the paper.

        Args:
            measurements (List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]): The measurements for setup.
            total (Optional[int]): Total number of records, if known.
        """
        total = self.set_total(total, measurements)
        self.total = total
        model_cliques = self.cliques = [m[3] for m in measurements]
        self.measurements = measurements
        self.model = ParticleModel(
            self.domain,
            embedding=self.embedding,
            n_particles=self.n_particles,
            data_init=self.data_init,
        )
        (
            self.centers,
            self.center_mappings,
        ) = self.embedding.get_centers_of_embedding(model_cliques)
        self.adjust_for_compression = {
            clique: self.embedding.adjust_for_compression(clique)
            for clique in model_cliques
        }
        self.dimsX = self.embedding.get_dims(model_cliques)

        self.processed_measurements = []
        self.quantized_measurements = []
        logging.info("Starting projection step.")
        processed_measurements = self.projection_step(measurements)
        quantized_measurements, _ = self.deterministic_quantization(
            processed_measurements
        )
        self.processed_measurements = processed_measurements
        self.quantized_measurements = quantized_measurements
        logging.info("Projection step completed.")

    def set_total(
        self,
        total: Optional[int],
        measurements: List[
            Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]
        ],
    ) -> int:
        """
        Determines or validates the total number of records based on measurements.

        Args:
            total (Optional[int]): The provided total number of records.
            measurements (List[Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]]): Measurements to use
            for determining the total.

        Returns:
            int: The determined or validated total number of records.
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, noise, _ in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]
                if np.allclose(Q.T.dot(v), o):
                    variances = np.append(variances, noise**2 * np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
            if estimates.size == 0:
                total = 1
            else:
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(1, estimate)
        return total

    def projection_step(
        self,
        measurements: List[
            Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]
        ],
        reuse:bool=False
    ) -> List[Tuple[torch.Tensor, float, Tuple[str, ...]]]:
        """
        Projection step. This method transforms the finite signed measures into probability measures by minimizing sliced
        1-Wasserstein distance.

        Args:
            measurements (List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]): The measurements.

        Returns:
            List[Tuple[torch.Tensor, float, Tuple[str, ...]]]: preprocessed measurements
        """
        if not reuse:
            self.measurements_processed = []
        already_measured = [m[3] for m in self.measurements_processed]

        for _, y, noise, proj in measurements:
            if proj in already_measured:
                continue
            ynorm = y / self.total
            if self.adjust_for_compression[proj] is not None:
                ynorm = ynorm[self.adjust_for_compression[proj].cpu().numpy()]
            if proj in self.yprobs:
                yprobnorm = self.yprobs[proj]
            else:
                # Initialization: set negative weights to zero and normalize
                y_new = np.maximum(ynorm, 0)
                y_new /= y_new.sum()
                yprobnorm = torch.tensor(
                    y_new, dtype=torch.float32, device=self.device
                )

                yprobnorm = self.optimize_u(
                    yprobnorm,
                    torch.tensor(
                        ynorm, dtype=torch.float32, device=self.device
                    ),
                    self.centers[proj],
                    lr=0.1,
                    num_projections=self.num_projections_proj,
                    step=self.scheduler_step_proj,
                    gamma=self.scheduler_gamma_proj,
                    num_iterations=self.iters_proj,
                )
                self.yprobs[proj] = yprobnorm
            m = (yprobnorm, noise, proj)
            self.measurements_processed.append(m)
        return self.measurements_processed

    def optimize_u(
        self,
        initial_u: torch.Tensor,
        v: torch.Tensor,
        X: torch.Tensor,
        lr: float = 0.1,
        num_iterations: int = 5000,
        num_projections: int = 100,
        step: int = 100,
        gamma: float = 0.8,
    ) -> torch.Tensor:
        """
        Optimizes the tensor 'u' using gradient descent to minimize the sliced 1-Wasserstein distance to 'v'.

        Args:
            initial_u (torch.Tensor): The initial tensor 'u'.
            v (torch.Tensor): The target tensor 'v'.
            X (torch.Tensor): The matrix used for sliced Wasserstein distance.
            lr (float, optional): Learning rate. Defaults to 0.1.
            num_iterations (int, optional): Number of iterations for optimization. Defaults to 5000.
            num_projections (int, optional): Number of projections for sliced Wasserstein distance. Defaults to 100.
            step (int, optional): Step size for learning rate scheduler. Defaults to 100.
            gamma (float, optional): Multiplicative factor for learning rate decay. Defaults to 0.8.

        Returns:
            torch.Tensor: The optimized tensor 'u'.
        """
        u = initial_u.clone()
        params = torch.log(u + 1e-9)
        params.requires_grad = True
        optimizer = optim.Adam([params], lr=lr)
        step_size = step
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            # Compute the sliced 1-Wasserstein distance
            SW1 = self.sliced_one_wasserstein_distance(
                torch.softmax(params, dim=0),
                v,
                X,
                num_projections=num_projections,
            )
            SW1.backward()

            optimizer.step()
            scheduler.step()

        return torch.softmax(params, dim=0).detach()

    def sliced_one_wasserstein_distance(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        X: torch.Tensor,
        num_projections: int = 5,
    ) -> float:
        """
        Computes the sliced 1-Wasserstein (SW1) distance between two distributions 'u' and 'v'.
        We leverage closed-form of the 1-Wasserstein distance with Vallender's identity
        and approximate the SW1 with random projections.

        Args:
            u (torch.Tensor): The first distribution.
            v (torch.Tensor): The second distribution.
            X (torch.Tensor): Data matrix for projections.
            num_projections (int, optional): Number of random projections. Defaults to 5.

        Returns:
            float: The sliced Wasserstein distance.
        """
        _, d = X.shape

        # Generate random normalized directions: [num_projections, d]
        directions = torch.randn(
            num_projections, d, device=self.device, dtype=torch.float32
        )
        norm = directions.norm(dim=1, keepdim=True) + 1e-9
        directions /= norm

        # Project X onto random directions: [n, num_projections]
        X_proj = torch.mm(X - 1 / 2, directions.t())
        X_sorted, indices_X = torch.sort(X_proj, dim=0)

        # Sort projections and get the indices
        u_sorted = u[indices_X]

        v_sorted = v[indices_X]

        distances = torch.diff(X_sorted, dim=0)

        # Compute W1 using Vallender's identity
        cu = torch.cumsum(u_sorted, dim=0)[:-1, :]
        cv = torch.cumsum(v_sorted, dim=0)[:-1, :]
        c_diff = cu - cv

        # Average across random projections to compute the SW1
        W1 = torch.sum(torch.abs(c_diff) * distances, dim=0)
        return W1.mean()

    def deterministic_quantization(
        self,
        measurements: List[Tuple[torch.Tensor, float, Tuple[str, ...]]],
    ) -> Tuple[
        Dict[Tuple[str, ...], torch.Tensor], Dict[Tuple[str, ...], torch.Tensor]
    ]:
        """
        This method applies deterministic quantization to the processed measurements by repeating rows
        of the center matrices according to the assigned probabilities. This helps in generating a quantized
        version of the measurements, which is essential for further processing in the inference mechanism.

        Returns:
            Tuple[Dict[Tuple[str, ...], torch.Tensor], Dict[Tuple[str, ...], torch.Tensor]]: tuple containing two dicts.
            The first dictionary maps projections to their quantized measurements as tensors.
            The second dictionary holds the probability values associated with each projection.
        """
        yprobvals = {}
        measurements_quantized = {}
        for yprob, _, proj in measurements:
            measurements_quantized[proj] = self.repeat_rows(
                self.centers[proj], yprob, self.n_particles
            )
            yprobvals[proj] = yprob
        return measurements_quantized, yprobvals

    def repeat_rows(
        self, matrix: torch.Tensor, weights: torch.Tensor, N: int
    ) -> torch.Tensor:
        """
        Repeats rows of a matrix based on weights.

        Args:
            matrix (torch.Tensor): The matrix whose rows are to be repeated.
            weights (torch.Tensor): The weights determining the repetition of rows.
            N (int): The total number of rows in the resulting matrix.

        Returns:
            torch.Tensor: The matrix with repeated rows.
        """
        # Calculate the number of repetitions for each row based on its weight
        num_repeats = (weights / torch.sum(weights) * N).int()

        new_matrix = matrix.repeat_interleave(num_repeats, dim=0)

        # If there are any leftover rows due to rounding, append them randomly
        leftover = N - new_matrix.shape[0]
        if leftover > 0:
            indices = torch.multinomial(weights, leftover, replacement=True)
            new_matrix = torch.cat([new_matrix, matrix[indices]], dim=0)

        return new_matrix

    def particle_gradient_descent(self) -> float:
        """
        Performs particle gradient descent to optimize the particle model.

        Returns:
            float: The loss value after optimization.
        """

        logging.info("Starting particle gradient descent optimization.")

        # Set up batching, optimizer and scheduler
        self.history_particles = []
        model = self.model
        paramsX = model.X.requires_grad_(True)
        iters = self.iters
        lrX = self.lr
        # Convert dictionary keys to a list for batching
        proj_keys = [proj for yprob, noise, proj in self.processed_measurements]
        # Set batch_size based on hp['batch_size']
        batch_size = len(proj_keys) if self.batch_size == 0 else self.batch_size
        # If random masking, use sparse version of Adam
        if self.p_mask > 0:
            optimizerX = torch.optim.SparseAdam([paramsX], lr=lrX)
        else:
            optimizerX = torch.optim.Adam([paramsX], lr=lrX)
        schedulerX = torch.optim.lr_scheduler.StepLR(
            optimizerX,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )

        # Optimization step, minimize SW2 between empirical and private marginal distributions
        total_loss = 0
        losses = {}
        proj_indices = list(range(len(proj_keys)))
        for epoch in range(1, iters + 1):  # Iterate epochs
            # Shuffle the keys for random batches
            random.shuffle(proj_indices)  # Shuffle the indices
            paramsX.grad = torch.zeros_like(paramsX).to(self.device)

            for i in range(0, len(proj_keys), batch_size):  # Iterate batches
                batch_start = i
                batch_end = min(i + batch_size, len(proj_keys))
                batch_keys_indices = proj_indices[batch_start:batch_end]
                batch_keys = [proj_keys[ind] for ind in batch_keys_indices]

                # Loss and gradient for domain-specific constraints
                if self.constraint_regularizer is not None:
                    new_paramsX = paramsX.clone().detach().requires_grad_(True)
                    (
                        grad,
                        _,
                    ) = self.constraint_regularizer.get_gradient_and_loss(
                        new_paramsX
                    )
                    grad_mat = grad * (self.scale_reg * 100)
                else:
                    grad_mat = torch.zeros_like(paramsX)

                # Loss and gradient for SW2
                for proj in batch_keys:  # Iterate measurements
                    Yarr = self.quantized_measurements[proj]
                    dimX = self.dimsX[proj]
                    X_selected = paramsX.detach()[:, dimX]

                    (
                        loss,
                        grad_X_batch,
                    ) = self.sliced_two_wasserstein_squared_distance_and_gradient(
                        X_selected,
                        Yarr,
                        n_projections=self.num_projections,
                    )
                    total_loss += loss
                    grad_mat[:, dimX] += grad_X_batch.detach()

                    losses[proj] = loss

                # Mask tensors
                if self.p_mask > 0:
                    paramsX.grad = self.mask_tensor(
                        grad_mat, self.p_mask
                    ).to_sparse()

                optimizerX.step()  # update parameters
                projected_X = self.embedding.projection_manifold(
                    paramsX.detach()
                )
                paramsX.data.copy_(projected_X)
                self.model.X = paramsX.detach()

            schedulerX.step()

            total_loss = np.sum(
                np.array([tloss for key, tloss in losses.items()])
            )

            if self.eval_particles:
                synth = self.model.synthetic_data()
                self.history_particles.append(synth.df)

            self.history.append(total_loss)
            logging.info(
                "Epoch %d/%d: Total loss = %f", epoch, iters, total_loss
            )

        logging.info("Particle gradient descent optimization completed.")
        return self.history[-1]

    def sliced_two_wasserstein_squared_distance_and_gradient(
        self, X: torch.Tensor, Y: torch.Tensor, n_projections: int = 100
    ) -> Tuple[float, torch.Tensor]:
        """
        Computes the sliced 2-Wasserstein squared distance and its gradient. First, efficiently computes
        the W2 distance by running a sorting algorithm. Second, approximates the SW2 with random projections.

        Args:
            X (torch.Tensor): The first set of samples.
            Y (torch.Tensor): The second set of samples.
            n_projections (int, optional): Number of projections for slicing. Defaults to 100.

        Returns:
            Tuple[float, torch.Tensor]: The sliced Wasserstein 2 squared distance and the gradient.
        """
        dim = X.size(1)
        batch_size = X.size(0)

        projections = self.sample_random_directions(dim, n_projections)
        proj_X = torch.mm(X, projections.t())
        proj_Y = torch.mm(Y, projections.t())

        X_sorted, indices_X = torch.sort(proj_X, dim=0)
        Y_sorted, _ = torch.sort(proj_Y, dim=0)

        swd_ret = torch.mean(torch.mean((X_sorted - Y_sorted) ** 2, dim=0))

        diffs = X_sorted - Y_sorted
        expanded_diffs = diffs.unsqueeze(2).expand(-1, -1, dim)
        expanded_theta = projections.unsqueeze(0).expand(batch_size, -1, -1)
        grads = 2 * expanded_diffs * expanded_theta
        grads_reconstructed = torch.zeros_like(grads)
        for d in range(dim):
            grads_reconstructed[:, :, d].scatter_(0, indices_X, grads[:, :, d])
        return swd_ret.item(), grads_reconstructed.mean(1) * 100

    def sample_random_directions(
        self, d: int, num_directions: int = 100
    ) -> torch.Tensor:
        """
        Samples random directions for projections.

        Args:
            d (int): The dimension of each direction vector.
            num_directions (int, optional): Number of directions to sample. Defaults to 100.

        Returns:
            torch.Tensor: The matrix of sampled directions.
        """
        theta = torch.randn(num_directions, d).to(self.device)
        norm = torch.norm(theta, dim=1, keepdim=True) + 1e-9
        return theta / norm

    def mask_tensor(self, Y: torch.Tensor, p: float) -> torch.Tensor:
        """
        Masks a given percentage of a tensor.

        Args:
            Y (torch.Tensor): The tensor to be masked.
            p (float): Percentage of the tensor to mask.

        Returns:
            torch.Tensor: The masked tensor.
        """
        # assert Y.is_cuda, "Input tensor should be on the GPU."

        # Compute the number of elements to mask
        num_to_mask = int(Y.numel() * p / 100)

        # Choose random indices to mask without repetition.
        mask_indices = torch.randperm(Y.numel(), device=Y.device)[:num_to_mask]

        # Create a flat view of the tensor to easily index into it and mask the chosen indices
        Y.view(-1)[mask_indices] = 0  # or set to another value

        return Y
