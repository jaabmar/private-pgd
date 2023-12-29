from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from mechanisms.ektelo_matrix import Identity
from mechanisms.mechanism import Mechanism
from mechanisms.privacy_calibrator import gaussian_mech

if TYPE_CHECKING:
    from inference.dataset import Dataset
    from inference.pgm.inference import FactoredInference
    from inference.privpgd.inference import AdvancedSlicedInference


class KWay(Mechanism):
    def __init__(
        self,
        epsilon: float,
        delta: float = 0.00001,
        degree: int = 2,
        bounded: bool = True,
        prng: np.random = np.random,
    ):
        """
        Initializes the K-Way mechanism for differential privacy.

        Args:
            epsilon (float): Differential privacy parameter epsilon.
            delta (float): Differential privacy parameter delta. Defaults to 0.00001.
            degree (int): Degree of the K-Way mechanism. Defaults to 2.
            bounded (bool): Indicates if the privacy definition is bounded or unbounded. Defaults to True.
            prng (np.random): Pseudo Random Number Generator. Defaults to np.random.
        """
        super(KWay, self).__init__(
            epsilon=epsilon, delta=delta, bounded=bounded, prng=prng
        )
        self.k = degree

    def run(
        self,
        data: "Dataset",
        engine: Union["FactoredInference", "AdvancedSlicedInference"],
        workload: List[Tuple[str, ...]],
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
        sigma = (
            gaussian_mech(self.epsilon, self.delta)["sigma"]
            * np.sqrt(len(workload))
            * self.marginal_sensitivity
        )
        total = data.records if self.bounded else None
        measurements = []
        for cl in workload:
            Q = Identity(data.domain.size(cl))
            x = data.project(cl).datavector()
            y = x + np.random.normal(loc=0, scale=sigma, size=x.size)
            measurements.append((Q, y, sigma, cl))

        est, loss = engine.estimate(measurements, total)
        print("Generating Data...")
        synth = est.synthetic_data(records)
        return synth, loss
