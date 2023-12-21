from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from inference.domain import Domain

from inference.torch_factor import Factor


class Embedding:
    def __init__(
        self,
        domain: "Domain",
        base_domain: Optional["Domain"] = None,
        supports: Optional[Dict[str, List[bool]]] = None,
    ):
        """
        Initializes an Embedding object used for discretizing and embedding data points.

        Args:
            domain (Domain): The domain associated with the embedding.
            base_domain (Optional[Domain]): The base domain, defaults to the same as `domain` if not provided.
            supports (Optional[Dict[str, List[bool]]]): Specifies the supports for each attribute in the domain.
        """
        self.domain = domain
        self.embedding = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bijection = {}
        self.bijection_length = 0
        self.centers = {}
        if base_domain is None:
            self.base_domain = self.domain
        else:
            self.base_domain = base_domain
        if supports is None:
            self.supports = {}
            for attr in self.base_domain.config:
                self.supports[attr] = [
                    True for _ in range(self.base_domain.config[attr])
                ]
        else:
            self.supports = supports

        self.modified = {}
        for attr in self.base_domain.config:
            self.modified[attr] = sum(self.supports[attr]) < len(
                self.supports[attr]
            )

        start_index = 0

        for attr, shape in zip(self.base_domain.attrs, self.base_domain.shape):
            # Check if the embedding type is 's'
            if self.base_domain.embedding_type[attr] == "s":
                # Create the bins for the hypercube
                self.embedding[attr] = np.linspace(0, 1, shape + 1)
                # Compute and store the centers of the bins
                self.centers[attr] = torch.tensor(
                    (
                        (self.embedding[attr][1:] + self.embedding[attr][:-1])
                        / 2
                    ),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(1)

                # Create a bijection from attribute to integer (0 to K-1)
                self.bijection[attr] = list(range(start_index, start_index + 1))
                start_index += 1
                self.bijection_length += 1

    def get_embedding(
        self, attrs: Union[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Retrieves the embedding for the specified attributes.

        Args:
            attrs (Union[str, List[str]]): The attributes for which to get the embedding.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping attributes to their embeddings.
        """
        if isinstance(attrs, str):
            attrs = [attrs]
        return {attr: self.embedding.get(attr, None) for attr in attrs}

    def get_bijection(
        self, attrs: Union[str, List[str]]
    ) -> Dict[str, List[int]]:
        """
        Retrieves the bijection mapping for the specified attributes.

        Args:
            attrs (Union[str, List[str]]): The attributes for which to get the bijection.

        Returns:
            Dict[str, List[int]]: A dictionary mapping attributes to their bijections.
        """
        if isinstance(attrs, str):
            attrs = [attrs]
        return {attr: self.bijection.get(attr, None) for attr in attrs}

    def compute_distance_matrix_to_center(
        self,
        attrs: List[str],
        X: torch.Tensor,
        centers: torch.Tensor,
        p: int = 5,
    ) -> torch.Tensor:
        """
        Computes the distance matrix between data points X and centers for the specified attributes.

        Args:
            attrs (List[str]): The attributes to consider.
            X (torch.Tensor): The data points.
            centers (torch.Tensor): The centers to compute distances to.
            p (int): The norm degree for distance calculation.

        Returns:
            torch.Tensor: The computed distance matrix.
        """
        center = centers[attrs]

        # Concatenate indices from bijection for selected attributes
        dimsX = [index for attr in attrs for index in self.bijection[attr]]

        # Convert dimsX to a PyTorch tensor
        dimsX = torch.tensor(dimsX, device=self.device)

        # Extract the selected dimensions from X
        X_selected = X[:, dimsX]

        # Compute the differences between X_selected and centers
        diff = X_selected[:, None, :] - center[None, :, :]

        # Compute the distance matrix using p-norm
        distance_matrix = torch.norm(diff, p=p, dim=-1)

        return distance_matrix

    def get_dims(
        self, cliques: List[Tuple[str, ...]]
    ) -> Dict[Tuple[str, ...], torch.Tensor]:
        """
        Retrieves the dimensions for the specified cliques.

        Args:
            cliques (List[Tuple[str, ...]]): The cliques to get dimensions for.

        Returns:
            Dict[Tuple[str, ...], torch.Tensor]: A dictionary mapping cliques to their dimensions.
        """
        dims = {}
        for attrs in cliques:
            # Concatenate indices from bijection for selected attributes
            dimsX = [index for attr in attrs for index in self.bijection[attr]]
            # Convert dimsX to a PyTorch tensor
            dims[attrs] = torch.tensor(dimsX, device=self.device)

        return dims

    def compute_distance_matrix(
        self, X: torch.Tensor, Y: torch.Tensor, p: float = float("inf")
    ) -> torch.Tensor:
        """
        Computes the distance matrix between two sets of data points X and Y.

        Args:
            X (torch.Tensor): The first set of data points.
            Y (torch.Tensor): The second set of data points.
            p (float): The norm degree for distance calculation.

        Returns:
            torch.Tensor: The computed distance matrix.
        """
        diff = X[:, None, :] - Y[None, :, :]
        distance_matrix = torch.norm(diff, p=p, dim=-1)
        return distance_matrix

    def compute_distance_matrices_centers(
        self,
        cliques: List[Tuple[str, ...]],
        centers: Dict[Tuple[str, ...], torch.Tensor],
        mappings: Dict[Tuple[str, ...], Dict[str, int]],
        p: int = 5,
    ) -> Dict[Tuple[str, ...], torch.Tensor]:
        """
        Computes distance matrices between centers for specified cliques.

        Args:
            cliques (List[Tuple[str, ...]]): The cliques for which to compute distance matrices.
            centers (Dict[Tuple[str, ...], torch.Tensor]): The centers for the cliques.
            mappings (Dict[Tuple[str, ...], Dict[str, int]]): The mappings for the cliques.
            p (int): The norm degree for distance calculation.

        Returns:
            Dict[Tuple[str, ...], torch.Tensor]: A dictionary mapping cliques to their distance matrices.
        """
        distance_matrices = {}
        for attrs in cliques:
            center = centers[attrs]
            mapping = mappings[attrs]
            dims = [index for attr in attrs for index in mapping[attr]]
            diff = center[:, None, dims] - center[None, :, dims]
            distance_matrices[attrs] = torch.norm(diff, p=p, dim=-1)
        return distance_matrices

    def get_centers_of_embedding(
        self, cliques: Optional[List[Tuple[str, ...]]] = None
    ) -> Tuple[
        Dict[Tuple[str, ...], torch.Tensor],
        Dict[Tuple[str, ...], Dict[str, int]],
    ]:
        """
        Retrieves the centers of the embeddings for the specified cliques.

        Args:
            cliques (Optional[List[Tuple[str, ...]]]): The cliques for which to get the centers.
                                                    If None, defaults to all attributes in the domain.

        Returns:
            Tuple[Dict[Tuple[str, ...], torch.Tensor], Dict[Tuple[str, ...], Dict[str, int]]]: A tuple containing:
                - The first dictionary maps cliques to their centers.
                - The second dictionary provides attribute mappings for each clique.
        """
        centers = {}
        mappings = {}
        vector_dict = {
            key: [
                row
                for row, is_true in zip(matrix, self.supports[key])
                if is_true
            ]
            for key, matrix in self.centers.items()
        }
        if not cliques:
            cliques = [list(self.domain.attrs)]

        for clique in cliques:
            combinations = list(product(*[vector_dict[key] for key in clique]))
            temp = torch.stack(
                [torch.concatenate(comb) for comb in combinations]
            )
            centers[clique] = temp.type(torch.float32).to(self.device)
            mapping = {}
            t = 0
            for key in clique:
                dimkey = len(self.bijection[key])
                mapping[key] = list(range(t, t + dimkey))
                t += dimkey
            mappings[clique] = mapping
        return centers, mappings

    def project(
        self,
        newDomain: "Domain",
        X: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> "Factor":
        """
        Projects the data points X onto a new domain.

        Args:
            newDomain (Domain): The new domain to project onto.
            X (torch.Tensor): The data points to be projected.
            weights (Optional[torch.Tensor]): The weights for the data points. If None, equal weights are used.

        Returns:
            Factor: A Factor object representing the projected data.
        """
        attrs = newDomain.attrs
        shapes = [newDomain.config[attr] for attr in attrs]
        n = X.shape[0]

        if weights is None:
            weights = torch.ones((n)) / n

        X_disc = self.discretize(X, attrs)

        values = self.count_vector_occurrences(X_disc, shapes, weights=weights)

        return Factor(newDomain, values)

    def discretize(
        self,
        X: torch.Tensor,
        attrs: Union[str, List[str]],
        p: float = float("inf"),
    ) -> torch.Tensor:
        """
        Discretizes the data points X for the specified attributes.

        Args:
            X (torch.Tensor): The data points to be discretized.
            attrs (Union[str, List[str]]): The attributes to consider for discretization.
            p (float): The norm degree used for discretization.

        Returns:
            torch.Tensor: The discretized data points.
        """
        n = X.shape[0]

        X_disc = torch.zeros(n, len(attrs), dtype=int)

        for j, attr in enumerate(attrs):
            diff = (
                X[:, self.bijection[attr], None]
                - self.centers[attr][self.supports[attr], :][None, :]
            )
            dist = torch.norm(diff, p=p, dim=-1)
            min_dist_idx = torch.argmin(dist, dim=-1)
            X_disc[:, j] = min_dist_idx

        return X_disc

    def embedd(
        self, df: pd.DataFrame, attrs: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Embeds the given DataFrame according to the specified attributes.

        Args:
            df (pd.DataFrame): The DataFrame to embed.
            attrs (Optional[List[str]]): The attributes to use for embedding. If None, uses all attributes in the domain.

        Returns:
            torch.Tensor: The embedded DataFrame.
        """
        if attrs is None:
            attrs = self.domain.attrs

        n = df.shape[0]

        X_emb = torch.zeros(
            n,
            sum([len(self.bijection[attr]) for attr in attrs]),
            dtype=torch.float32,
        )
        location = 0
        for _, attr in enumerate(attrs):
            new_location = len(self.bijection[attr]) + location
            X_emb[:, location:new_location] = self.centers[attr][df[attr], :]
            location = new_location
        return X_emb

    def count_vector_occurrences(
        self,
        X: torch.Tensor,
        shape: List[int],
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Counts the occurrences of each vector in X, considering the given shape and weights.

        Args:
            X (torch.Tensor): The data vectors to count.
            shape (List[int]): The shape of each vector dimension.
            weights (Optional[torch.Tensor]): The weights for each data vector. If None, equal weights are used.

        Returns:
            torch.Tensor: A tensor representing the count of each vector occurrence.
        """
        n = X.shape[0]
        if weights is None:
            weights = torch.ones((n)) / n

        # rescale the weights so that they sum to n
        rescaled_weights = n * weights.detach().cpu()
        # Convert each row in X to a single index
        indices = torch.zeros(X.shape[0], dtype=int)
        multiplier = 1
        for i in reversed(range(X.shape[1])):
            indices += X[:, i] * multiplier
            multiplier *= shape[i]

        # Count occurrences of each index
        counts = torch.bincount(
            indices, weights=rescaled_weights, minlength=multiplier
        )

        return counts

    def adjust_for_compression(
        self, attrs: List[str]
    ) -> Optional[torch.Tensor]:
        """
        Adjusts for MST (Minimum Spanning Tree) by considering the specified attributes.

        Args:
            attrs (List[str]): The attributes to consider for adjustment.

        Returns:
            Optional[torch.Tensor]: A tensor representing the adjusted values. Returns None if no adjustment is needed.
        """
        shapes = [self.domain.config[key] for key in attrs]
        adjust_shapes = [
            sum(self.supports[attr]) < len(self.supports[attr])
            for attr in attrs
        ]

        if not any(adjust_shapes):
            return None

        ranges = [range(s) for s in shapes]

        v = list(product(*ranges))
        h = [
            not any(
                b and v_el[i] == shapes[i] - 1
                for i, b in enumerate(adjust_shapes)
            )
            for v_el in v
        ]

        return torch.tensor(h, device=self.device)

    def projection_manifold(self, X):
        X.clamp_(0, 1)
        return X
