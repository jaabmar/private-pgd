import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import softmax

from inference.pgm.graphical_model import GraphicalModel
from inference.pgm.inference import FactoredInference
from mechanisms.mechanism import Mechanism

if TYPE_CHECKING:
    from inference.dataset import Dataset


class MWEM(Mechanism):
    def __init__(
        self,
        epsilon: float,
        delta: float = 0.00001,
        rounds: int = 100,
        max_model_size: float = 1000,
        bounded: bool = True,
        prng: np.random = np.random,
    ):
        """
        Initializes the MWEM mechanism for differential privacy.

        Args:
            epsilon (float): Differential privacy parameter epsilon.
            delta (float): Differential privacy parameter delta. Defaults to 0.00001.
            rounds (int): The number of rounds for the mechanism. Defaults to 100.
            max_model_size (float): Maximum model size in MBytes. Defaults to 1000.
            bounded (bool): Indicates if the privacy definition is bounded or unbounded. Defaults to True.
            prng (np.random): Pseudo Random Number Generator. Defaults to np.random.
        """
        super(MWEM, self).__init__(
            epsilon=epsilon, delta=delta, bounded=bounded, prng=prng
        )
        self.rounds = rounds
        self.max_model_size = max_model_size

    def worst_approximated(
        self,
        workload_answers: Dict[Tuple[str, ...], np.ndarray],
        est: "GraphicalModel",
        workload: List[Tuple[str, ...]],
        eps: float,
        penalty: bool = True,
    ) -> Tuple[str, ...]:
        """
        Selects the worst approximated candidate using the exponential mechanism.

        Args:
            workload_answers (Dict[Tuple[str, ...], np.ndarray]): True answers for the queries.
            est (GraphicalModel): The estimated model used for projection.
            workload (List[Tuple[str, ...]]): Workload of queries.
            eps (float): Epsilon value for differential privacy.
            penalty (bool): Flag to apply penalty. Defaults to True.

        Returns:
            Tuple[str, ...]: The selected worst approximated candidate.
        """
        errors = np.array([])
        for cl in workload:
            x = workload_answers[cl]
            bias = est.domain.size(cl) if penalty else 0
            xest = est.project(cl).datavector()
            errors = np.append(errors, np.abs(x - xest).sum() - bias)

        prob = softmax(0.5 * eps / self.sensitivity * (errors - errors.max()))
        key = np.random.choice(len(errors), p=prob)
        return workload[key]

    def run(
        self,
        data: "Dataset",
        engine: "FactoredInference",
        workload: List[Tuple[str, ...]],
        alpha: float = 0.9,
        records: Optional[int] = None,
    ) -> Tuple["Dataset", float]:
        """
        Runs the MWEM mechanism to generate a synthetic dataset.

        Args:
            data (Dataset): The original dataset.
            workload (Optional[List[Tuple[str, ...]]]): A list of queries as tuples of attributes. Defaults to None.
            engine (FactoredInference): The inference engine used for estimation.
            alpha (float): Alpha parameter for controlling noise. Defaults to 0.9.
            records(Optional[int]): Number of samples of the generated dataset. Defaults to None, same as original dataset.

        Returns:
            Tuple[Dataset, float]: The synthetic dataset and the associated loss.
        """
        rounds = min(len(workload), self.rounds)
        rho = self.rho
        rho_per_round = rho / rounds
        sigma = (
            np.sqrt(0.5 / (alpha * rho_per_round)) * self.marginal_sensitivity
        )
        exp_eps = np.sqrt(8 * (1 - alpha) * rho_per_round)
        domain = data.domain
        total = data.records if self.bounded else None

        def size(cliques):
            return GraphicalModel(domain, cliques).size * 8 / 2**20

        workload_answers = {
            cl: data.project(cl).datavector() for cl in workload
        }

        measurements = []
        est, _ = engine.estimate(measurements, total)
        cliques = []
        for i in range(1, rounds + 1):
            candidates = [
                cl
                for cl in workload
                if size(cliques + [cl]) <= self.max_model_size * i / rounds
            ]
            ax = self.worst_approximated(
                workload_answers, est, candidates, exp_eps
            )
            logging.info(
                "Round %d, Selected %s, Model Size (MB) %f",
                i,
                ax,
                est.size * 8 / 2**20,
            )

            n = domain.size(ax)
            x = data.project(ax).datavector()

            y = x + np.random.normal(loc=0, scale=sigma, size=n)
            Q = sparse.eye(n)
            measurements.append((Q, y, 1.0, ax))
            est, loss = engine.estimate(measurements, total)
            cliques.append(ax)

        print("Generating Data...")
        return est.synthetic_data(records), loss
