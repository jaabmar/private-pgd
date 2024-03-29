from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import numpy as np
import ot
import pandas as pd

from inference.embedding import Embedding

if TYPE_CHECKING:
    from inference.dataset import Dataset


class Evaluator:
    def __init__(
        self,
        data: "Dataset",
        synth: "Dataset",
        workload: List[Tuple[str, ...]],
        undo_compress_fn: Callable[["Dataset"], "Dataset"] = lambda x: x,
    ):
        """
        Initializes the Evaluator for comparing synthetic data with actual data.

        Args:
            data (Dataset): The actual dataset.
            synth (Dataset): The synthetic dataset generated by a model.
            workload (List[Tuple[str, ...]]): A list of projections representing different queries.
            undo_compress_fn (Callable[[Dataset], Dataset]): A function to undo the compression on the synthetic data.
                                                            Defaults to an identity function.
        """
        self.data = data
        self.synth = synth
        self.workload = workload
        self.undo_compress_fn = undo_compress_fn

        self.embedding = Embedding(self.data.domain)

        self.Xemb = {}
        self.weightspr = {}

        self.dataproj = data
        self.synthproj = synth

    def update_synth(self, synth: "Dataset") -> None:
        """
        Updates the synthetic dataset.

        Args:
            synth (Dataset): The new synthetic dataset.
        """
        self.synth = self.undo_compress_fn(synth)

    def set_compression(
        self, undo_compress_fn: Callable[["Dataset"], "Dataset"] = lambda x: x
    ) -> None:
        """
        Sets the function to undo compression on the synthetic data.

        Args:
            undo_compress_fn (Callable[[Dataset], Dataset]): A function to undo the compression. Defaults to an identity.
        """
        self.undo_compress_fn = undo_compress_fn

    def evaluate(
        self,
        print_output: bool = False,
        name: str = "",
        # use_wandb: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluates the synthetic data against the actual data using various metrics.

        Args:
            step (Optional[int]): The current step in an iterative process, if applicable.
            print_output (bool): If True, prints the evaluation results.
            name (str): A prefix for metric names.
            use_wandb (bool): If True, logs the metrics to Weights & Biases.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: A dictionary of distances calculated using different metrics.
        """
        res = {}
        res.update(self.evaluate_l1(name=name))
        res.update(self.evaluate_l2(name=name))
        eval_w, dists = self.evaluate_w(name=name)

        res.update(eval_w)
        # if use_wandb:
        #     import wandb

        #     wandb.log(res)
        if print_output:
            print(f"the evaluation shows: {res}")
        return res, dists

    def evaluate_l1(self, name: str = "") -> Dict[str, float]:
        """
        Evaluates the L1 distance between the actual and synthetic data for each projection in the workload.

        Args:
            name (str): A prefix for metric names.

        Returns:
            Dict[str, float]: A dictionary containing the mean and sum of L1 distances for each projection.
        """
        errors = []
        for proj in self.workload:
            X = self.data.project(proj).datavector()
            Y = self.synth.project(proj).datavector()
            e = np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
            errors.append(e)
        return {
            f"{name}l1_avg": np.mean(errors),
            f"{name}l1_max": np.max(errors),
        }

    def evaluate_l2(self, name: str = "") -> Dict[str, float]:
        """
        Evaluates the L2 distance between the actual and synthetic data for each projection in the workload.

        Args:
            name (str): A prefix for metric names.

        Returns:
            Dict[str, float]: A dictionary containing the mean and sum of L2 distances for each projection.
        """
        errors = []
        for proj in self.workload:
            X = self.data.project(proj).datavector()
            Y = self.synth.project(proj).datavector()
            e = np.linalg.norm(X / X.sum() - Y / Y.sum(), 2)
            errors.append(e)
        return {
            f"{name}l2_avg": np.mean(errors),
            f"{name}l2_max": np.max(errors),
        }

    def evaluate_w(
        self, name: str = ""
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, ...], float]]:
        """
        Evaluates the Wasserstein distance between the actual and synthetic data for each projection in the workload.

        Args:
            name (str): A prefix for metric names.

        Returns:
            Tuple[Dict[str, float], Dict[Tuple[str, ...], float]]: A tuple where the first element is a dictionary
            containing the mean, sum, and maximum of Wasserstein distances for each projection, and the second element
            is a dictionary of Wasserstein distances for each individual projection.
        """
        errors = []
        dicterrors = {}
        for proj in self.workload:
            self.dataproj = self.data.project(proj)

            X, weightsprt = self.compress(
                self.dataproj.df.values, self.dataproj.weights
            )
            self.weightspr[proj] = abs(weightsprt) / sum(abs(weightsprt))
            lproj = list(proj)
            self.Xemb[proj] = (
                self.embedding.embedd(pd.DataFrame(X, columns=lproj), proj)
                .detach()
                .cpu()
                .numpy()
            )

            self.synthproj = self.synth.project(proj)
            Xpub, weightspub = self.compress(
                self.synthproj.df.values, self.synthproj.weights
            )
            weightspr = self.weightspr[proj]
            weightspub = abs(weightspub) / sum(abs(weightspub))
            lproj = list(proj)
            Xemb = self.Xemb[proj]
            Xpubemb = (
                self.embedding.embedd(pd.DataFrame(Xpub, columns=lproj), proj)
                .detach()
                .cpu()
                .numpy()
            )
            errors.append(
                self.efficient_weighted_swd(
                    Xemb, weightspr, Xpubemb, weightspub
                )
            )
            dicterrors[proj] = errors[-1]
        return {
            f"{name}w_avg": np.mean(errors),
            f"{name}w_sum": np.sum(errors),
            f"{name}w_max": max(errors),
        }, dicterrors

    def efficient_weighted_swd(
        self,
        Xemb: np.ndarray,
        weightspr: np.ndarray,
        Xpubemb: np.ndarray,
        weightspub: np.ndarray,
        m: int = 100,
    ) -> float:
        """
        Efficiently computes the sliced Wasserstein distance between two weighted empirical distributions.

        Args:
            Xemb (np.ndarray): Samples from the first distribution.
            weightspr (np.ndarray): Weights for the samples from the first distribution.
            Xpubemb (np.ndarray): Samples from the second distribution.
            weightspub (np.ndarray): Weights for the samples from the second distribution.
            m (int): Number of random projections used in the computation.

        Returns:
            float: The average sliced Wasserstein distance over the random projections.
        """

        # Ensure weights are normalized
        weightspr = weightspr / np.sum(weightspr)
        weightspub = weightspub / np.sum(weightspub)

        distances = []

        for _ in range(m):
            # Generate a random 1D projection
            proj = np.random.normal(size=Xemb.shape[1])
            proj /= np.linalg.norm(proj)

            # Project the data
            Xemb_proj = Xemb @ proj
            Xpubemb_proj = Xpubemb @ proj

            # Compute 1D Wasserstein distance using ot's efficient method
            dist = ot.wasserstein_1d(
                Xemb_proj, Xpubemb_proj, weightspr, weightspub
            )
            distances.append(dist)

        # Return the average Wasserstein distance across all projections
        return np.mean(distances)

    def compress(
        self, X: Union[np.ndarray, pd.DataFrame], weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compresses the data points X and their weights.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data points to be compressed.
            weights (np.ndarray): The weights associated with each data point.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the compressed data points and their weights.
        """
        df = pd.DataFrame(X)
        df["weights"] = weights
        # group by all columns in X, sum the weights, and reset the index
        df = df.groupby(list(range(df.shape[1] - 1))).sum().reset_index()

        X_prime = df[df.columns[:-1]].values
        weights_prime = df[df.columns[-1]].values
        return X_prime, weights_prime
