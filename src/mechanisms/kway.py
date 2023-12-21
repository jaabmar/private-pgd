from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from mechanisms.ektelo_matrix import Identity
from mechanisms.mechanism import Mechanism
from mechanisms.privacy_calibrator import gaussian_mech
from mechanisms.utils_mechanisms import downward_closure

if TYPE_CHECKING:
    from inference.dataset import Dataset
    from inference.pgm.inference import FactoredInference
    from inference.privpgd.inference import AdvancedSlicedInference


class KWay(Mechanism):
    def __init__(
        self,
        hp: Dict[str, Any],
        bounded: bool = True,
        prng: np.random = np.random,
    ):
        """
        Initializes the K-Way mechanism for differential privacy.

        Args:
            hp (Dict[str, Any]): A dictionary of hyperparameters containing epsilon, delta, and degree.
            prng (np.random): Pseudo Random Number Generator. Defaults to None.
            bounded (bool): Privacy definition (bounded vs unbounded DP).
        """
        super(KWay, self).__init__(
            epsilon=hp["epsilon"], delta=hp["delta"], bounded=bounded, prng=prng
        )
        self.k = hp["degree"]
        self.hp = hp

    def run(
        self,
        data: "Dataset",
        workload: List[Tuple[str, ...]],
        engine: Union["FactoredInference", "AdvancedSlicedInference"],
        records: Optional[int] = None,
    ) -> Tuple["Dataset", float]:
        """
        Runs the K-Way mechanism to generate a synthetic dataset.

        Args:
            data (Dataset): The original dataset.
            workload (List[Tuple[str, ...]]): A list of queries as tuples of attributes.
            engine (Union[FactoredInference, AdvancedSlicedInference]): The inference engine used for estimation.
            records(Optional[int]): Number of samples of the generated dataset. Defaults to None, same as original dataset.

        Returns:
            Tuple[Dataset, float]: The synthetic dataset and the associated loss.
        """
        delta_f = 2 * len(workload)
        b = delta_f / self.epsilon
        marginal_sensitivity = np.sqrt(2) if self.bounded else 1.0
        sigma = gaussian_mech(self.epsilon, self.delta)["sigma"] * np.sqrt(
            len(workload)
        )
        total = data.records if self.bounded else None
        measurements = []
        for cl in workload:
            Q = Identity(data.domain.size(cl))
            x = data.project(cl).datavector()
            y = x + np.random.normal(
                loc=0, scale=marginal_sensitivity * sigma, size=x.size
            )
            measurements.append((Q, y, b, cl))

        est, loss = engine.estimate(measurements, total)
        print("Generating Data...")
        synth = est.synthetic_data(records)
        return synth, loss

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
