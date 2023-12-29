import json

import numpy as np
import pandas as pd


class DataHandler:
    def __init__(self, df=None):
        if df is not None:
            self.df = df
        self.inverse_mapping = {}
        self.bin_edges = {}
        self.domain = {}

    def forward(self, k):
        df_transformed = pd.DataFrame()

        for column in self.df.columns:
            if pd.api.types.is_bool_dtype(self.df[column]):
                df_transformed[column] = np.array(self.df[column], dtype=int)
                self.domain[column] = 2
                self.inverse_mapping[column] = [False, True]
                print(df_transformed)
                continue

            if pd.api.types.is_numeric_dtype(self.df[column]):
                # Check if column is of integer type and has a range less than k

                if pd.api.types.is_integer_dtype(self.df[column]):
                    data_range = (
                        self.df[column].max() - self.df[column].min() + 1
                    )
                    print(f"is integer with range {data_range}")
                    if data_range < k:
                        k = data_range

                # Equal spaced binning
                df_transformed[column], bins = pd.cut(
                    self.df[column],
                    k,
                    labels=False,
                    retbins=True,
                    duplicates="drop",
                )

                # Store the bin edges for back transformation
                self.bin_edges[column] = bins

                # Store the center of each bin for back transformation
                bin_centers = [
                    (bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)
                ]
                self.inverse_mapping[column] = bin_centers

                # Set domain for the column
                self.domain[column] = k

            else:
                print(self.df[column][0])
                df_transformed[column], unique_index = pd.factorize(
                    self.df[column]
                )

                # df_transformed[column], unique_index = pd.factorize(self.df[column])
                self.inverse_mapping[column] = unique_index

                # Set domain for the column
                self.domain[column] = len(unique_index)

        return df_transformed, self.domain, self.inverse_mapping

    def backward(self, df_transformed):
        df_original = pd.DataFrame()

        for column in df_transformed.columns:
            if isinstance(self.inverse_mapping[column], pd.Index) or isinstance(
                self.inverse_mapping[column], np.ndarray
            ):
                df_original[column] = self.inverse_mapping[column][
                    df_transformed[column]
                ].values
            else:
                bin_medians = self.inverse_mapping[column]
                df_original[column] = df_transformed[column].map(
                    lambda x: bin_medians[x]
                    if (x < len(bin_medians) and x >= 0)
                    else np.nan
                )

        return df_original

    def save(self, filename):
        """Saves the object attributes to a JSON file."""
        attributes = {
            "inverse_mapping": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.inverse_mapping.items()
            },
            "bin_edges": {k: v.tolist() for k, v in self.bin_edges.items()},
            "domain": self.domain,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(attributes, f)

    @classmethod
    def load(cls, filename):
        """Loads the object attributes from a JSON file and returns a DataHandler object."""
        with open(filename, "r", encoding="utf-8") as f:
            attributes = json.load(f)

        handler = cls()
        handler.inverse_mapping = {
            k: pd.Index(v) if isinstance(v, list) else v
            for k, v in attributes["inverse_mapping"].items()
        }
        handler.bin_edges = {
            k: np.array(v) for k, v in attributes["bin_edges"].items()
        }
        handler.domain = attributes["domain"]

        return handler
