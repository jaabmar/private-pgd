import pickle
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch

from inference.dataset import Dataset
from inference.domain import Domain
from inference.embedding import Embedding
from inference.factor import Factor


class ParticleModel:
    def __init__(
        self,
        domain: "Domain",
        embedding: Optional["Embedding"] = None,
        n_particles: int = 500,
        data_init: Optional["Dataset"] = None,
    ):
        """
        Initializes a ParticleModel for probabilistic modeling using particles.

        Args:
            domain (Domain): The domain associated with the particle model.
            embedding (Optional[Embedding]): An embedding object. If None, a new embedding is created. Defaults to None.
            n_particles (int): The number of particles to use. Defaults to 500.
            data_init (Optional[Dataset]): A dataset for initializing particles. If None, particles are initialized
                                        randomly. Defaults to None.
        """
        self.domain = domain

        if embedding is None:
            self.embedding = Embedding(domain)
        else:
            self.embedding = embedding

        if not data_init:
            self.n_particles = n_particles

            self.X = torch.rand(
                n_particles,
                self.embedding.bijection_length,
                dtype=torch.float32,
                device=self.embedding.device,
            )  # particles randomly drawn from the d-dimensional hypercube
            self.w = torch.full(
                (n_particles,),
                1 / n_particles,
                dtype=torch.float32,
                device=self.embedding.device,
            )  # weights are 1/n by default
        else:
            self.n_particles = data_init.df.shape[0]

            if data_init.weights is not None:
                self.w = torch.tensor(
                    data_init.weights / sum(data_init.weights),
                    dtype=torch.float32,
                    device=self.embedding.device,
                )
            else:
                self.w = torch.full(
                    (self.n_particles,),
                    1 / self.n_particles,
                    dtype=torch.float32,
                    device=self.embedding.device,
                )  # weights are 1/n by default
            self.X = self.embedding.embedd(data_init.df)

    @staticmethod
    def save(model: "ParticleModel", path: str) -> None:
        """
        Saves the ParticleModel to a file.

        Args:
            model (ParticleModel): The ParticleModel instance to save.
            path (str): The file path to save the model to.
        """
        pickle.dump(model, open(path, "wb"))

    @staticmethod
    def load(path: str) -> "ParticleModel":
        """
        Loads a ParticleModel from a file.

        Args:
            path (str): The file path from which to load the model.

        Returns:
            ParticleModel: The loaded ParticleModel instance.
        """
        return pickle.load(open(path, "rb"))

    @staticmethod
    def base_norm(v: torch.Tensor) -> torch.Tensor:
        """
        Computes the L2 norm of a given tensor.

        Args:
            v (torch.Tensor): The tensor to compute the norm for.

        Returns:
            torch.Tensor: The L2 norm of the tensor.
        """
        return torch.norm(v, p=2, dim=-1)

    def compute_distance_matrix(
        self,
        attrs: Optional[List[str]] = None,
        other: Optional["ParticleModel"] = None,
    ) -> torch.Tensor:
        """
        Computes the distance matrix between particles.

        Args:
            attrs (Optional[List[str]]): A subset of attributes to consider. If None, uses all attributes.
            other (Optional[ParticleModel]): Another ParticleModel instance to compare with. If None, compares with itself.

        Returns:
            torch.Tensor: The computed distance matrix.
        """
        if attrs is None:
            attrs = self.domain.attrs
        if other is None:
            other = self

        return self.embedding.compute_distance_matrix(attrs, self.X, other.X)

    def project(self, attrs: Union[List[str], Tuple[str, ...]]) -> "Factor":
        """
        Projects the particle model onto a subset of attributes, computing the marginal distribution.

        Args:
            attrs (Union[List[str], Tuple[str, ...]]): A subset of attributes in the domain to project onto.

        Returns:
            Factor: A Factor object representing the marginal distribution.
        """
        attrs_shape = [self.domain.config[attr] for attr in attrs]
        newDomain = Domain(attrs, attrs_shape)

        return self.embedding.project(newDomain, self.X, self.w)

    def datavector(self, flatten: bool = True) -> torch.Tensor:
        """
        Materializes the explicit representation of the distribution as a data vector.

        Args:
            flatten (bool): If True, flattens the resulting data vector. Defaults to True.

        Returns:
            torch.Tensor: The data vector representing the distribution.
        """
        return self.project(self.domain.attrs).datavector()

    def synthetic_data(
        self, rows: Optional[int] = None, method: str = "round"
    ) -> Dataset:
        """
        Generates synthetic tabular data from the particle model.

        Args:
            rows (Optional[int]): The number of rows in the synthetic dataset. If None, uses the total count of the model.
            Defaults to None.
            method (str): The method to generate data, either 'round' or 'sample'. Defaults to 'round'.

        Returns:
            Dataset: The generated synthetic dataset.
        """
        X_disc = self.embedding.discretize(self.X, self.domain.attrs)
        weights = self.w.detach().cpu().numpy()
        df = pd.DataFrame(
            X_disc.detach().cpu().numpy(), columns=self.domain.attrs
        )
        return Dataset(df, self.domain, weights=weights)
