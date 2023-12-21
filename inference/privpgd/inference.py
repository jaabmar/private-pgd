import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse
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
        N: int,
        hp: Dict[str, Any],
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
        self.N = N
        self.hp = hp
        self.embedding = embedding if embedding else Embedding(domain)
        self.constraint_regularizer = constraint_regularizer
        self.device = hp["device"]
        self.iters = hp["iters"]
        self.yprobs = {}

        self.n_particles = (
            self.hp["n_particles"]
            if self.hp["n_particles"] > 0
            else min(
                self.N * self.hp["n_times_particles"], self.hp["max_particles"]
            )
        )
        self.model = None
        self.measurements = []
        self.cliques = []
        self.centers = {}
        self.center_mappings = {}
        self.adjust_for_compression = {}
        self.dimsX = {}
        self.processed_measurements = []

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
        self._setup(measurements)
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
    ):
        """
        Sets up the estimation process by initializing various components based on the given measurements.
        Includes projection step from the paper.

        Args:
            measurements (List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]): The measurements for setup.
        """
        model_cliques = self.cliques = [m[3] for m in measurements]
        self.measurements = measurements
        self.model = ParticleModel(
            self.domain,
            embedding=self.embedding,
            n_particles=self.n_particles,
            data_init=self.hp["data_init"],
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

        self.projection_step(measurements)

    def projection_step(
        self,
        measurements: List[
            Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]
        ],
    ):
        """
        Projection step. This method transforms the finite signed measures into probability measures by minimizing sliced
        1-Wasserstein distance.

        Args:
            measurements (List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]): The measurements.
        """
        for _, y, noise, proj in measurements:
            ynorm = y / self.N
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
                    num_projections=200,
                    step=100,
                    gamma=0.8,
                    num_iterations=1750,
                )
                self.yprobs[proj] = yprobnorm
            m = (yprobnorm, noise, proj)
            self.processed_measurements.append(m)

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
        directions /= directions.norm(dim=1, keepdim=True) + 1e-9

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

    def particle_gradient_descent(self) -> float:
        """
        Performs particle gradient descent to optimize the particle model.

        Returns:
            float: The loss value after optimization.
        """

        # Set up batching, optimizer and scheduler
        model = self.model
        hp = self.hp
        paramsX = model.X.requires_grad_(True)
        iters = self.iters
        lrX = hp["lr"]
        # Convert dictionary keys to a list for batching
        proj_keys = [proj for yprob, noise, proj in self.processed_measurements]
        # Set batch_size based on hp['batch_size']
        batch_size = (
            len(proj_keys) if hp["batch_size"] == 0 else hp["batch_size"]
        )
        # If random masking, use sparse version of Adam
        if self.hp["p_mask"] > 0:
            optimizerX = torch.optim.SparseAdam([paramsX], lr=lrX)
        else:
            optimizerX = torch.optim.Adam([paramsX], lr=lrX)
        schedulerX = torch.optim.lr_scheduler.StepLR(
            optimizerX,
            step_size=hp["scheduler_step_size"],
            gamma=hp["scheduler_gamma"],
        )

        # Quantization
        new_measurements, _ = self.deterministic_quantization()

        # Optimization step, minimize SW2 between empirical and private distributions
        total_loss = 0
        losses = {}
        proj_indices = list(range(len(proj_keys)))
        for _ in range(1, iters + 1):  # Iterate epochs
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
                    new_paramsX = torch.tensor(paramsX.data).requires_grad_(
                        True
                    )
                    (
                        grad,
                        _,
                    ) = self.constraint_regularizer.get_gradient_and_loss(
                        new_paramsX, hp
                    )
                    grad_mat = grad * (hp["scale_reg"] * 100)
                else:
                    grad_mat = torch.zeros_like(paramsX)

                # Loss and gradient for SW2
                for proj in batch_keys:  # Iterate measurements
                    Yarr = new_measurements[proj]
                    dimX = self.dimsX[proj]
                    X_selected = paramsX.detach()[:, dimX]

                    (
                        loss,
                        grad_X_batch,
                    ) = self.sliced_two_wasserstein_squared_distance_and_gradient(
                        X_selected,
                        Yarr,
                        n_projections=self.hp["num_projections"],
                    )
                    total_loss += loss
                    grad_mat[:, dimX] += grad_X_batch.detach()

                    losses[proj] = loss

                # Mask tensors
                if self.hp["p_mask"] > 0:
                    paramsX.grad = self.mask_tensor(
                        grad_mat, self.hp["p_mask"]
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

            return total_loss

    def deterministic_quantization(
        self,
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
        quantized_measurements = {}
        for yprob, _, proj in self.processed_measurements:
            quantized_measurements[proj] = self.repeat_rows(
                self.centers[proj], yprob, self.n_particles
            )
            yprobvals[proj] = yprob
        return quantized_measurements, yprobvals

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
