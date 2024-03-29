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
        hp: Dict[str, Any],
        structural_zeros: Dict[str, List[Tuple]] = {},
        elim_order: Optional[List[str]] = None,
    ):
        """
        Initializes the FactoredInference class for learning a GraphicalModel from noisy measurements on a data distribution.

        Args:
            domain (Domain): The domain information.
            hp (Dict[str, Any]): Hyperparameters.
            structural_zeros (Dict[str, List[Tuple]]): An encoding of known zeros in the distribution.
            elim_order (Optional[List[str]]): An elimination order for the JunctionTree algorithm.
        """
        self.domain = domain
        self.metric = "L2"
        self.iters = hp["iters"] if "iters" in hp else 3000
        self.warm_start = hp["warm_start"] if "warm_start" in hp else False
        self.stepsize = hp["lr"] if "lr" in hp else 1.0
        self.elim_order = elim_order

        self.embedding = Embedding(domain, base_domain=None, supports=None)
        self.device = self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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
    ) -> Tuple["GraphicalModel", float]:
        """
        Estimates a GraphicalModel from the given measurements.

        Args:
            measurements (List[Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]]): Measurements for
            estimation.
            total (Optional[int]): Total number of records, if known.

        Returns:
            Tuple[GraphicalModel, float]: The estimated GraphicalModel and the loss value.
        """

        measurements = self.fix_measurements(measurements)
        self._setup(measurements, total)
        loss = self.mirror_descent()
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
        assert type(measurements) is list, (
            "measurements must be a list, given " + measurements
        )
        assert all(
            len(m) == 4 for m in measurements
        ), "each measurement must be a 4-tuple (Q, y, noise,proj)"
        ans = []
        for Q, y, noise, proj in measurements:
            assert (
                Q is None or Q.shape[0] == y.size
            ), "shapes of Q and y are not compatible"
            if type(proj) is list:
                proj = tuple(proj)
            if type(proj) is not tuple:
                proj = (proj,)
            if Q is None:
                Q = sparse.eye(self.domain.size(proj))
            assert np.isscalar(
                noise
            ), "noise must be a real value, given " + str(noise)
            assert all(a in self.domain for a in proj), (
                str(proj) + " not contained in domain"
            )
            assert Q.shape[1] == self.domain.size(
                proj
            ), "shapes of Q and proj are not compatible"
            ans.append((Q, y, noise, proj))
        return ans

    def _setup(
        self,
        measurements: List[
            Tuple[np.ndarray, np.ndarray, float, Union[str, Tuple[str, ...]]]
        ],
        total: Optional[int] = None,
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
        if self.warm_start and self.model:
            model.potentials.combine(self.model.potentials)
        self.model = model
        # group the measurements into model cliques
        model_cliques = self.model.cliques
        self.groups = defaultdict(lambda: [])
        self.fixed_measurements = []
        for Q, y, noise, proj in measurements:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
            if isinstance(Q, np.ndarray):
                Q = torch.tensor(Q, dtype=torch.float32, device=self.device)
            elif sparse.issparse(Q):
                Q = Q.tocoo()
                idx = torch.LongTensor(np.array([Q.row, Q.col])).to(self.device)
                vals = torch.FloatTensor(Q.data).to(self.device)
                Q = torch.sparse_coo_tensor(idx, vals, device=self.device)
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

    def mirror_descent(
        self,
    ) -> float:
        """
        Performs the mirror descent algorithm to estimate the GraphicalModel.
        Returns:
            float: The loss value after optimization.
        """

        model = self.model
        theta = model.potentials
        mu = model.belief_propagation(theta)
        ans = self.marginal_loss(mu)
        if ans[0] == 0:
            return ans[0]
        if np.isscalar(self.stepsize):
            alpha = float(self.stepsize)
            stepsize = lambda t: alpha
        if self.stepsize is None:
            alpha = 1.0 / model.total**2
            stepsize = lambda t: 2.0 * alpha

        for t in range(1, self.iters + 1):
            omega, nu = theta, mu
            curr_loss, dL = ans
            alpha = stepsize(t)
            for _ in range(25):
                theta = omega - alpha * dL
                mu = model.belief_propagation(theta)
                ans = self.marginal_loss(mu)
                if curr_loss - ans[0] >= 0.5 * alpha * dL.dot(nu - mu):
                    break
                alpha *= 0.5

        model.potentials = theta
        model.marginals = mu
        self.model = model

        return ans[0]

    def marginal_loss(
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
                loss += 0.5 * (diff @ diff)
                grad = c * (Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)
