import json
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from inference.domain import Domain

class Dataset:
    def __init__(
        self,
        df: pd.DataFrame,
        domain: "Domain",
        weights: Optional[np.ndarray] = None,
    ):
        """
        Create a Dataset object.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the data.
            domain (Domain): A Domain object representing the domain of the dataset.
            weights (Optional[np.ndarray]): An array of weights for each row in the DataFrame. Defaults to None.

        Raises:
            AssertionError: If the domain attributes are not a subset of DataFrame columns,
                            or if the weights array does not match the number of rows in the DataFrame.
        """
        assert set(domain.attrs) <= set(
            df.columns
        ), "Data must contain domain attributes"
        assert (
            weights is None or df.shape[0] == weights.size
        ), "Weights array must match the number of rows in the DataFrame"

        self.domain = domain
        self.df = df.loc[:, domain.attrs]
        self.weights = (
            weights if weights is not None else np.ones(self.df.shape[0])
        )
        if weights is not None and sum(self.weights) < 1.5:
            self.weights = self.weights * self.df.shape[0]

    @staticmethod
    def synthetic(domain: "Domain", N: int) -> "Dataset":
        """
        Generate synthetic data conforming to the given domain.

        Args:
            domain (Domain): The domain object defining the dataset structure.
            N (int): The number of rows (individuals) in the synthetic dataset.

        Returns:
            Dataset: A Dataset object containing synthetic data.
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path: str, domain_path: str) -> "Dataset":
        """
        Load data into a Dataset object from a CSV file and domain JSON file.

        Args:
            path (str): Path to the CSV file containing data.
            domain_path (str): Path to the JSON file encoding the domain information.

        Returns:
            Dataset: A Dataset object loaded with the data from the CSV and domain files.
        """
        df = pd.read_csv(path)
        with open(domain_path, encoding="utf-8") as file:
            config = json.load(file)
        domain = Domain(
            list(config.keys()), list(config.values()), d=df.shape[1]
        )
        return Dataset(df, domain)

    def project(
        self, cols: Union[str, int, List[Union[str, int]]]
    ) -> "Dataset":
        """
        Project the dataset onto a subset of columns.

        Args:
            cols (Union[str, int, List[Union[str, int]]]): The columns to project onto.

        Returns:
            Dataset: A Dataset object containing data for the specified columns.
        """
        if isinstance(cols, (str, int)):
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols: Union[str, int, List[Union[str, int]]]) -> "Dataset":
        """
        Drop specified columns from the dataset.

        Args:
            cols (Union[str, int, List[Union[str, int]]]): Columns to be dropped.

        Returns:
            Dataset: A Dataset object with specified columns dropped.
        """
        proj = [c for c in self.domain.attrs if c not in cols]
        return self.project(proj)

    @property
    def records(self) -> int:
        """
        Get the number of records in the dataset.

        Returns:
            int: The number of records.
        """
        return self.df.shape[0]

    def datavector(self, flatten: bool = True) -> np.ndarray:
        """
        Return the database in vector-of-counts form.

        Args:
            flatten (bool): If True, the histogram is flattened into a 1D array. Defaults to True.

        Returns:
            np.ndarray: The dataset in vector-of-counts form.
        """
        bins = [range(n + 1) for n in self.domain.shape]
        histogram = np.histogramdd(self.df.values, bins, weights=self.weights)[
            0
        ]
        return histogram.flatten() if flatten else histogram
