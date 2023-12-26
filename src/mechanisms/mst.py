import itertools
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from disjoint_set import DisjointSet
from scipy import sparse

from inference.dataset import Dataset
from inference.domain import Domain
from mechanisms.mechanism import Mechanism

if TYPE_CHECKING:
    from inference.pgm.inference import FactoredInference
    from inference.privpgd.inference import AdvancedSlicedInference


class MST(Mechanism):
    def __init__(
        self,
        epsilon: float,
        delta: float = 0.00001,
        bounded: bool = True,
    ):
        """
        Initializes the MST mechanism for differential privacy.

        Args:
            epsilon (float): Differential privacy parameter epsilon.
            delta (float): Differential privacy parameter delta. Defaults to 0.00001.
            bounded (bool): Indicates if the privacy definition is bounded or unbounded. Defaults to True.
        """
        super(MST, self).__init__(epsilon=epsilon, delta=delta, bounded=bounded)

    def run(
        self,
        data: "Dataset",
        engine: Union["FactoredInference", "AdvancedSlicedInference"],
        workload: List[Tuple[str, ...]] = None,
        records: Optional[int] = None,
    ) -> Tuple["Dataset", float]:
        """
        Runs the MST mechanism to generate a synthetic dataset.

        Args:
            data (Dataset): The original dataset.
            engine (Union[FactoredInference,AdvancedSlicedInference]): The inference engine used for estimation.
            records(Optional[int]): Number of samples of the generated dataset. Defaults to None, same as original dataset.

        Returns:
            Tuple[Dataset, float]: The synthetic dataset and the associated loss.
        """
        rho = self.rho
        sigma = np.sqrt(3 / (2 * rho))
        total = data.records if self.bounded else None

        cliques_oneway = [(col,) for col in data.domain]
        log1 = self.measure(data=data, cliques=cliques_oneway, sigma=sigma)

        est, loss = engine.estimate(log1, total)
        cliques = self.select(
            est=est,
            data=data,
        )

        log2 = self.measure(data=data, cliques=cliques, sigma=sigma)
        est, loss = engine.estimate(log1 + log2, total)
        print("Generating Data...")
        synth = est.synthetic_data(records)
        return synth, loss

    def measure(
        self,
        data: "Dataset",
        cliques: List[Tuple[str, ...]],
        sigma: float,
        weights: Optional[np.ndarray] = None,
    ) -> List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]:
        """
        Measures the data with added Gaussian noise.

        Args:
            data (Dataset): The dataset to measure.
            cliques (List[Tuple[str, ...]]): A list of cliques to measure.
            sigma (float): Standard deviation of the Gaussian noise.
            weights (Optional[np.ndarray]): Weights for each clique. Defaults to None.

        Returns:
            List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]: A list of measurements.
        """
        if weights is None:
            weights = np.ones(len(cliques))
        weights /= np.linalg.norm(weights)
        measurements = []
        for proj, wgt in zip(cliques, weights):
            x = data.project(proj).datavector()
            y = x + np.random.normal(
                loc=0,
                scale=sigma / wgt * self.marginal_sensitivity,
                size=x.size,
            )
            Q = sparse.eye(x.size)
            measurements.append(
                (Q, y, sigma / wgt * self.marginal_sensitivity, proj)
            )
        return measurements

    def select(
        self,
        est: Union["FactoredInference", "AdvancedSlicedInference"],
        data: "Dataset",
        cliques: List[Tuple[str, ...]] = [],
    ) -> List[Tuple[str, ...]]:
        """
        Selects additional cliques based on estimation errors.

        Args:
            est (Union[FactoredInference,AdvancedSlicedInference]): The current estimation.
            data (Dataset): The dataset.
            cliques (List[Tuple[str, ...]]): The list of current cliques.
        Returns:
            List[Tuple[str, ...]]: The list of selected cliques.
        """
        weights, T, ds = {}, nx.Graph(), DisjointSet()
        T.add_nodes_from(data.domain.attrs)
        for e in cliques:
            T.add_edge(*e)
            ds.union(*e)

        candidates = list(itertools.combinations(data.domain.attrs, 2))

        for a, b in candidates:
            xhat = est.project([a, b]).datavector()
            x = data.project([a, b]).datavector()
            weights[a, b] = np.linalg.norm(x - xhat, 1)

        r = len(list(nx.connected_components(T)))
        epsilon = np.sqrt(8 * self.rho / (r - 1))
        for _ in range(r - 1):
            candidates = [e for e in candidates if not ds.connected(*e)]
            wgts = np.array([weights[e] for e in candidates])
            idx = self.exponential_mechanism(
                qualities=wgts,
                epsilon=epsilon,
            )
            e = candidates[idx]
            T.add_edge(*e)
            ds.union(*e)
        chosen_candidates = list(T.edges)
        return chosen_candidates

    def transform_data(
        self, data: "Dataset", supports: Dict[str, np.ndarray]
    ) -> "Dataset":
        """
        Transforms the data according to the supports.

        Args:
            data (Dataset): The dataset to transform.
            supports (Dict[str, np.ndarray]): The supports dictionary.

        Returns:
            Dataset: The transformed dataset.
        """
        df, newdom = data.df.copy(), {}
        for col in data.domain:
            support = supports[col]
            size = support.sum()
            newdom[col] = int(size)
            if size < support.size:
                newdom[col] += 1
            mapping = {}
            idx = 0
            for i in range(support.size):
                mapping[i] = size
                if support[i]:
                    mapping[i] = idx
                    idx += 1
            assert idx == size
            df[col] = df[col].map(mapping)

        newdom = Domain.fromdict(newdom)
        return Dataset(df, newdom)

    def reverse_data(
        self, data: "Dataset", supports: Dict[str, np.ndarray]
    ) -> "Dataset":
        """
        Reverses the data transformation.

        Args:
            data (Dataset): The dataset to reverse transform.
            supports (Dict[str, np.ndarray]): The supports dictionary.

        Returns:
            Dataset: The reversed transformed dataset.
        """
        df = data.df.copy()
        newdom = {}
        for col in data.domain:
            support = supports[col]
            mx = support.sum()
            newdom[col] = int(support.size)
            idx, extra = np.where(support)[0], np.where(~support)[0]
            mask = df[col] == mx
            if extra.size == 0:
                pass
            else:
                df.loc[mask, col] = np.random.choice(extra, mask.sum())
            df.loc[~mask, col] = idx[df.loc[~mask, col]]
        newdom = Domain.fromdict(newdom)
        return Dataset(df, newdom)

    def compress_domain(
        self,
        data: "Dataset",
        measurements: List[
            Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]
        ],
    ) -> Tuple[
        "Dataset",
        List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]],
        Callable,
        Dict[str, np.ndarray],
    ]:
        """
        Compresses the domain of the data based on measurements.

        Args:
            data (Dataset): The dataset to compress.
            measurements (List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]]): The measurements.

        Returns:
            Tuple[Dataset, List[Tuple[sparse.spmatrix, np.ndarray, float, Tuple[str, ...]]], Callable[[Dataset], Dataset],
            Dict[str, np.ndarray]]: The transformed dataset, new measurements, function to undo compression,
            and supports dictionary.
        """
        supports, new_measurements = {}, []
        for Q, y, sigma, proj in measurements:
            col = proj[0]
            sup = y >= 3 * sigma
            supports[col] = sup
            if supports[col].sum() == y.size:
                new_measurements.append((Q, y, sigma, proj))
            else:  # need to re-express measurement over the new domain
                y2 = np.append(y[sup], y[~sup].sum())
                I2 = np.ones(y2.size)
                I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
                y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
                I2 = sparse.diags(I2)
                new_measurements.append((I2, y2, sigma, proj))
        undo_compress_fn = lambda data: self.reverse_data(data, supports)
        return (
            self.transform_data(data, supports),
            new_measurements,
            undo_compress_fn,
            supports,
        )
