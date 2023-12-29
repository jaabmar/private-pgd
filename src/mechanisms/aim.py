import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import softmax

from inference.pgm.graphical_model import GraphicalModel
from inference.pgm.inference import FactoredInference
from mechanisms.ektelo_matrix import Identity
from mechanisms.mechanism import Mechanism
from mechanisms.utils_mechanisms import downward_closure

if TYPE_CHECKING:
    from inference.dataset import Dataset
    from inference.domain import Domain


class AIM(Mechanism):
    def __init__(
        self,
        epsilon: float,
        delta: float = 0.00001,
        rounds: Optional[int] = None,
        max_model_size: float = 1000,
        bounded: bool = True,
        prng: np.random = np.random,
    ):
        """
        Initializes the AIM mechanism for differential privacy.

        Args:
            epsilon (float): Differential privacy parameter epsilon.
            delta (float): Differential privacy parameter delta. Defaults to 0.00001.
            rounds (Optional[int]): The number of rounds for the mechanism. Defaults to None.
            max_model_size (float): Maximum model size in MBytes. Defaults to 1000.
            bounded (bool): Indicates if the privacy definition is bounded or unbounded. Defaults to True.
            prng (np.random): Pseudo Random Number Generator. Defaults to np.random.
        """
        super(AIM, self).__init__(
            epsilon=epsilon, delta=delta, bounded=bounded, prng=prng
        )
        self.rounds = rounds
        self.max_model_size = max_model_size

    def worst_approximated(
        self,
        answers: Dict[Tuple[str, ...], np.ndarray],
        est: "GraphicalModel",
        candidates: Dict[Tuple[str, ...], float],
        eps: float,
        sigma: float,
    ):
        """
        Selects the worst approximated candidate using the exponential mechanism.

        Args:
            answers (Dict[Tuple[str, ...], np.ndarray]): True answers for the queries.
            est (GraphicalModel): The estimated model used for projection.
            candidates (Dict[Tuple[str, ...], float]): Workload of queries.
            eps (float): Epsilon value for differential privacy.
            sigma (float): Fixed standard deviation of the noise.

        Returns:
            Tuple[str, ...]: The selected worst approximated candidate.
        """
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * est.domain.size(cl)
            xest = est.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)
        max_sensitivity = max(sensitivity.values())
        keys = list(errors.keys())
        errors = np.array([errors[key] for key in keys])
        prob = softmax(0.5 * eps / max_sensitivity * (errors - errors.max()))
        key = self.prng.choice(prob.size, p=prob)
        return keys[key]
        # return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(
        self,
        data: "Dataset",
        engine: "FactoredInference",
        workload: List[Tuple[str, ...]],
        records: Optional[int] = None,
    ) -> Tuple["Dataset", float]:
        """
        Runs the AIM mechanism to generate a synthetic dataset.

        Args:
            data (Dataset): The original dataset.
            workload (Optional[List[Tuple[str, ...]]]): A list of queries as tuples of attributes. Defaults to None.
            engine (FactoredInference): The inference engine used for estimation.
            records(Optional[int]): Number of samples of the generated dataset. Defaults to None, same as original dataset.

        Returns:
            Tuple[Dataset, float]: The synthetic dataset and the associated loss.
        """
        rounds = self.rounds or 16 * len(data.domain)
        sigma = (
            np.sqrt(rounds / (2 * 0.9 * self.rho)) * self.marginal_sensitivity
        )
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)
        total = data.records if self.bounded else None

        candidates = self.compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        # get all the oneway marginals and estimate everything over that
        oneway = [cl for cl in candidates if len(cl) == 1]
        measurements = []
        logging.info("Initial Sigma: %s", sigma)
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in oneway:
            # discretization
            x = data.project(cl).datavector()
            # noise addition
            y = x + self.gaussian_noise(sigma, x.size)
            # make identity matrix
            identity = Identity(y.size)
            measurements.append((identity, y, sigma, cl))

        est, _ = engine.estimate(measurements, total)
        # run over all other marginals
        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (
                0.5 / sigma**2 + 1.0 / 8 * epsilon**2
            ):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            size_limit = self.max_model_size * rho_used / self.rho
            small_candidates = self.filter_candidates(
                candidates,
                est,
                size_limit,
            )
            cl = self.worst_approximated(
                answers, est, small_candidates, epsilon, sigma
            )
            n = data.domain.size(cl)
            x = data.project(cl).datavector()

            y = x + self.gaussian_noise(sigma, n)
            Q = Identity(n)
            measurements.append((Q, y, sigma, cl))
            z = est.project(cl).datavector()
            est, _ = engine.estimate(measurements, total)
            w = est.project(cl).datavector()
            logging.info(
                "Selected %s, Size %d, Budget Used %f",
                cl,
                n,
                rho_used / self.rho,
            )

            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                logging.info(
                    "(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma: %s", sigma / 2
                )

                sigma /= 2
                epsilon *= 2

        logging.info("Generating Data...")
        est, loss = engine.estimate(measurements, total)
        return est.synthetic_data(records), loss

    def compile_workload(
        self, workload: List[Tuple[str, ...]]
    ) -> Dict[Tuple[str, ...], float]:
        """
        Compiles a workload into a dictionary with scores for each query.

        Args:
            workload (List[Tuple[str, ...]]): A list of queries as tuples of attribute names.

        Returns:
            Dict[Tuple[str, ...], float]: A dictionary where each key is a query (tuple of attributes)
            and the value is the score of the query.
        """

        def score(cl):
            return sum(len(set(cl) & set(ax)) for ax in workload)

        return {cl: score(cl) for cl in downward_closure(workload)}

    def hypothetical_model_size(
        self, domain: "Domain", cliques: List[Tuple[str, ...]]
    ) -> float:
        """
        Calculates the hypothetical size of a graphical model with the given domain and cliques.

        Args:
            domain (Domain): The domain of the dataset.
            cliques (List[Tuple[str, ...]]): A list of cliques for the graphical model.

        Returns:
            float: The size of the hypothetical model in megabytes.
        """
        model = GraphicalModel(domain, cliques)
        return model.size * 8 / 2**20

    def filter_candidates(
        self,
        candidates: Dict[Tuple[str, ...], float],
        model: "GraphicalModel",
        size_limit: float,
    ) -> Dict[Tuple[str, ...], float]:
        """
        Filters candidates based on the size limit and their presence in the free cliques of the model.

        Args:
            candidates (Dict[Tuple[str, ...], float]): A dictionary of candidates with their scores.
            model (GraphicalModel): The current graphical model.
            size_limit (float): The maximum allowed size for the model in megabytes.

        Returns:
            Dict[Tuple[str, ...], float]: A filtered dictionary of candidates.
        """
        ans = {}
        free_cliques = downward_closure(model.cliques)
        for cl in candidates:
            cond1 = (
                self.hypothetical_model_size(model.domain, model.cliques + [cl])
                <= size_limit
            )
            cond2 = cl in free_cliques
            if cond1 or cond2:
                ans[cl] = candidates[cl]
        return ans
