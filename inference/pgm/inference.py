from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import lsmr

from inference.callbacks import Logger
from inference.clique_vector import CliqueVector
from inference.embedding import Embedding
from inference.pgm.graphical_model import GraphicalModel
from inference.torch_factor import Factor

if TYPE_CHECKING:
    from inference.domain import Domain


class FactoredInference:
    def __init__(
        self,
        domain: "Domain",
        N: int,
        hp: Dict[str, Any],
        structural_zeros: Dict[str, List[Tuple]] = {},
        log: bool = False,
        elim_order: Optional[List[str]] = None,
    ):
        """
        Initializes the FactoredInference class for learning a GraphicalModel from noisy measurements on a data distribution.

        Args:
            domain (Domain): The domain information.
            N (int): Total number of records in the dataset.
            hp (Dict[str, Any]): Hyperparameters.
            structural_zeros (Dict[str, List[Tuple]]): An encoding of known zeros in the distribution.
            log (bool): Flag to log iterations of optimization.
            elim_order (Optional[List[str]]): An elimination order for the JunctionTree algorithm.
        """
        self.domain = domain
        self.N = N
        self.log = log
        self.hp = hp
        self.elim_order = elim_order

        if hp["inference_type"] == "pgm_euclid":
            self.metric = "L2"
        else:
            self.metric = "L1"

        self.iters = hp["iterspgm"]
        self.warm_start = hp["warm_start"]

        self.history = []
        self.embedding = Embedding(domain, base_domain=None, supports=None)
        self.reg = 0.01
        self.device = self.hp["device"]

        self.Factor = Factor
        self.structural_zeros = CliqueVector({})
        for cl in structural_zeros:
            dom = self.domain.project(cl)
            fact = structural_zeros[cl]
            self.structural_zeros[cl] = self.Factor.active(dom, fact)

        self.model = None
        self.groups = defaultdict(lambda: [])
        self.fixed_measurements = []

    def estimate(
        self,
        measurements: List[
            Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]
        ],
        total: Optional[int] = None,
        callback: Optional[Callable[[Any], None]] = None,
        options: Dict[str, Any] = {},
    ) -> Tuple["GraphicalModel", float]:
        """
        Estimates a GraphicalModel from the given measurements.

        Args:
            measurements (List[Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]]): Measurements for
            estimation.
            total (Optional[int]): Total number of records, if known.
            callback (Optional[Callable[[Any], None]]): Function to call after each iteration of optimization.
            options (Dict[str, Any]): Solver-specific options.

        Returns:
            Tuple[GraphicalModel, float]: The estimated GraphicalModel and the loss value.
        """

        measurements = self.fix_measurements(measurements)
        options["callback"] = callback
        self._setup(measurements, total)

        if callback is None and self.log:
            options["callback"] = Logger(self)
        if self.hp["descent_type"] == "GD":
            loss = self.gradient_descent(measurements, total, **options)
        else:
            loss = self.mirror_descent(measurements, total, **options)

        return self.model, loss

    def fix_measurements(
        self,
        measurements: List[
            Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]
        ],
    ) -> List[
        Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]
    ]:
        """
        Fixes and preprocesses measurements for consistency.

        Args:
            measurements (List[Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]]): The raw measurements.

        Returns:
            List[Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]]: The processed measurements.
        """
        ans = []
        for Q, y, noise, proj in measurements:
            if type(proj) is list:
                proj = tuple(proj)
            if type(proj) is not tuple:
                proj = (proj,)
            if Q is None:
                Q = sparse.eye(self.domain.size(proj))
            ans.append((Q, y, noise, proj))
        return ans

    def _setup(
        self,
        measurements: List[
            Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]
        ],
        total: Optional[int],
    ) -> None:
        """
        Sets up the inference process based on the measurements.

        Args:
            measurements (List[Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]]): Measurements for setup.
            total (Optional[int]): Total number of records, if known.
        """
        total = self.set_total(total, measurements)
        cliques = [m[3] for m in measurements]
        if self.structural_zeros is not None:
            cliques += list(self.structural_zeros.keys())
        # Set up the model
        model = GraphicalModel(
            self.domain, cliques, total, elimination_order=self.elim_order
        )
        model.potentials = CliqueVector.zeros(self.domain, model.cliques)
        model.potentials.combine(self.structural_zeros)
        if self.warm_start and hasattr(self, "model"):
            model.potentials.combine(self.model.potentials)
        self.model = model
        # group the measurements into model cliques
        model_cliques = self.model.cliques
        self.groups = defaultdict(lambda: [])
        self.fixed_measurements = []
        for Q, y, noise, proj in measurements:
            if isinstance(Q, np.ndarray):
                Q = torch.tensor(Q, dtype=torch.float32, device=self.device)
            elif sparse.issparse(Q):
                Q = Q.tocoo()
                idx = torch.LongTensor([Q.row, Q.col]).to(self.device)
                vals = torch.FloatTensor(Q.data).to(self.device)
                Q = torch.sparse.FloatTensor(idx, vals).to(self.device)
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
            m = (Q, y, noise, proj)
            self.fixed_measurements.append(m)
            for cl in sorted(model_cliques, key=model.domain.size):
                if set(proj) <= set(cl):
                    self.groups[cl].append(m)
                    break

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

    def gradient_descent(self) -> float:
        """
        Performs gradient descent algorithm to estimate the GraphicalModel.

        Returns:
            float: The loss value after optimization.
        """

        hp = self.hp
        model = self.model
        cliques, theta = model.cliques, model.potentials
        params = {}
        for clique in cliques:
            theta[clique].values.requires_grad = True
            params[clique] = theta[clique].values

        if hp["optimizerpgm"] == "Adam":
            optimizer = torch.optim.Adam(
                [params[key] for key in params], lr=hp["lrpgm"]
            )
        elif hp["optimizerpgm"] == "SGD":
            optimizer = torch.optim.SGD(
                [params[key] for key in params], momentum=0.9, lr=hp["lrpgm"]
            )
        elif hp["optimizerpgm"] == "RMSProp":
            optimizer = torch.optim.RMSprop(
                [params[key] for key in params], lr=hp["lrpgm"]
            )

        if hp["scheduler_step"] > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hp["scheduler_step_pgm"],
                gamma=hp["scheduler_gamma_pgm"],
            )
        else:
            scheduler = None  # No scheduler if scheduler_step is zero

        curr_loss = 0
        for _ in range(1, self.iters + 1):
            with torch.no_grad():
                mu = model.belief_propagation(theta)
                curr_loss, dL = self._marginal_loss(mu, metric=self.metric)

            for clique in cliques:
                params[clique].grad = dL[clique].values
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        model.potentials = theta
        self.model = model
        return curr_loss

    def mirror_descent(
        self,
        callback: Optional[Callable[[Any], None]] = None,
    ) -> float:
        """
        Performs the mirror descent algorithm to estimate the GraphicalModel.

        Args:
            callback (Optional[Callable[[Any], None]]): Function to call after each iteration of optimization.

        Returns:
            float: The loss value after optimization.
        """

        model = self.model
        _, theta = model.cliques, model.potentials
        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)
        if ans[0] == 0:
            return ans[0]
        stepsize = self.hp["lrpgm"]

        if np.isscalar(stepsize):
            alpha = float(stepsize)
            stepsize = lambda t: alpha
        if stepsize is None:
            alpha = 1.0 / self.model.total**2
            stepsize = lambda t: 2.0 * alpha

        for t in range(1, self.iters + 1):
            if callback is not None:
                callback(mu)
            omega, nu = theta, mu
            curr_loss, dL = ans
            alpha = stepsize(t)
            for _ in range(25):
                theta = omega - alpha * dL
                mu = model.belief_propagation(theta)
                ans = self._marginal_loss(mu)
                if curr_loss - ans[0] >= 0.5 * alpha * dL.dot(nu - mu):
                    break
                alpha *= 0.5

        model.potentials = theta
        model.marginals = mu
        self.model = model

        return ans[0]

    def _marginal_loss(
        self,
        marginals: Dict[str, "Factor"],
        metric: Optional[Union[str, Callable]] = None,
    ) -> Tuple[float, "CliqueVector"]:
        """
        Computes the loss and gradient for given marginals.

        Args:
            marginals (Dict[str, Factor]): Dictionary of marginals.
            metric (Optional[Union[str, Callable[[Dict[str, Factor]], Tuple[float, CliqueVector]]]]): Metric for
            loss computation.

        Returns:
            Tuple[float, CliqueVector]: The loss value and the gradient.
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {}

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                c = 1.0 / noise
                mu2 = mu.project(proj)
                x = mu2.torch_datavector()
                diff = c * (Q @ x - y)
                if metric == "L1":
                    loss += abs(diff).sum()
                    sign = (
                        diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                    )
                    grad = c * (Q.T @ sign)
                else:
                    loss += 0.5 * (diff @ diff)
                    grad = c * (Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)

    def _marginal_loss_md(
        self,
        marginals: Dict[str, "Factor"],
        metric: Optional[Union[str, Callable]] = None,
    ) -> Tuple[float, "CliqueVector"]:
        """
        Computes the loss and gradient for given marginals using mirror descent.

        Args:
            marginals (Dict[str, Factor]): Dictionary of marginals.
            metric (Optional[Union[str, Callable[[Dict[str, Factor]], Tuple[float, CliqueVector]]]]): Metric for
            loss computation.

        Returns:
            Tuple[float, CliqueVector]: The loss value and the gradient.
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {}

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                Q = Q.to_dense()
                c = 1.0 / noise
                mu2 = mu.project(proj)
                x = mu2.datavector()
                diff = c * (Q @ x - y)
                if metric == "L1":
                    loss += abs(diff).sum()
                    sign = (
                        diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                    )
                    grad = c * (Q.T @ sign)
                else:
                    loss += 0.5 * (diff @ diff)
                    grad = c * (Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)
