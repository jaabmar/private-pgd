from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ektelo_matrix import Identity
from mechanism import Mechanism
from privacy_calibrator import gaussian_mech
from utils_mechanisms import downward_closure

from inference.dataset import Dataset
from inference.pgm.graphical_model import GraphicalModel
from inference.pgm.inference import FactoredInference
from inference.privpgd.inference import AdvancedSlicedInference
from inference.privpgd.particle_model import ParticleModel


class KWay(Mechanism):
    def __init__(self, hp: Dict[str, Any], prng: Optional[Any] = None):
        """
        Initializes the K-Way mechanism for differential privacy.

        Args:
            hp (Dict[str, Any]): A dictionary of hyperparameters containing epsilon, delta, and degree.
            prng (Optional[Any]): Pseudo Random Number Generator. Defaults to None.
        """
        super(KWay, self).__init__(hp["epsilon"], hp["delta"])
        self.epsilon = hp["epsilon"]
        self.delta = hp["delta"]
        self.k = hp["degree"]
        self.hp = hp

    def modify_workload(
        self, workload: List[Tuple[str, ...]]
    ) -> Dict[Tuple[str, ...], float]:
        """
        Modifies the workload for K-Way mechanism.

        Args:
            workload (List[Tuple[str, ...]]): A list of queries as tuples of attributes.

        Returns:
            Dict[Tuple[str, ...], float]: A dictionary with scores for each candidate in the workload.
        """

        def score(cl):
            return sum(len(set(cl) & set(ax)) for ax in workload)

        return {cl: score(cl) for cl in downward_closure(workload)}

    def worst_approximated(
        self,
        candidates: Dict[Tuple[str, ...], float],
        answers: Dict[Tuple[str, ...], np.ndarray],
        model: Union["GraphicalModel", "ParticleModel"],
        eps: float,
        sigma: float,
    ) -> Tuple[str, ...]:
        """
        Selects the worst approximated candidate using the exponential mechanism.

        Args:
            candidates (Dict[Tuple[str, ...], float]): Candidates with their scores.
            answers (Dict[Tuple[str, ...], np.ndarray]): True answers for the queries.
            model (Union[GraphicalModel, ParticleModel]): The graphical model used for projection.
            eps (float): Epsilon value for differential privacy.
            sigma (float): Standard deviation for the Gaussian noise.

        Returns:
            Tuple[str, ...]: The selected worst approximated candidate.
        """
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)
        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(
        self,
        data: "Dataset",
        workload: List[Tuple[str, ...]],
        engine: Union["FactoredInference", "AdvancedSlicedInference"],
    ) -> Tuple["Dataset", float]:
        """
        Runs the K-Way mechanism to generate a synthetic dataset.

        Args:
            data (Dataset): The original dataset.
            workload (List[Tuple[str, ...]]): A list of queries as tuples of attributes.
            engine (Union[FactoredInference, AdvancedSlicedInference]): The inference engine used for estimation.

        Returns:
            Tuple[Dataset, float]: The synthetic dataset and the associated loss.
        """
        delta_f = 2 * len(workload)
        # Calculate the scale parameter for Laplace noise
        b = delta_f / self.hp["epsilon"]
        sigma = (
            gaussian_mech(self.hp["epsilon"], self.hp["delta"])["sigma"]
            * np.sqrt(len(workload))
            * np.sqrt(2)
        )
        measurements = []
        for cl in workload:
            Q = Identity(data.domain.size(cl))
            x = data.project(cl).datavector()
            y = x + np.random.normal(loc=0, scale=sigma, size=x.size)
            measurements.append((Q, y, b, cl))

        model, loss = engine.estimate(measurements)
        synth = model.synthetic_data()
        return synth, loss
