from typing import Any, Dict, List

import numpy as np

from inference.dataset import Dataset
from inference.domain import Domain
from inference.torch_factor import Factor


class CliqueVector(dict):
    """
    A convenience class for arithmetic over concatenated vectors of marginals and potentials.
    These vectors are represented as a dictionary mapping cliques (subsets of attributes)
    to marginals/potentials (Factor objects).
    """

    def __init__(self, dictionary: Dict[Any, "Factor"]) -> None:
        super().__init__(dictionary)
        self.dictionary = dictionary

    @staticmethod
    def zeros(domain: "Domain", cliques: List) -> "CliqueVector":
        """Creates a CliqueVector with zero-valued Factors."""
        return CliqueVector(
            {cl: Factor.zeros(domain.project(cl)) for cl in cliques}
        )

    @staticmethod
    def ones(domain: "Domain", cliques: List) -> "CliqueVector":
        """Creates a CliqueVector with one-valued Factors."""
        return CliqueVector(
            {cl: Factor.ones(domain.project(cl)) for cl in cliques}
        )

    @staticmethod
    def uniform(domain: "Domain", cliques: List) -> "CliqueVector":
        """Creates a CliqueVector with uniformly distributed Factors."""
        return CliqueVector(
            {cl: Factor.uniform(domain.project(cl)) for cl in cliques}
        )

    @staticmethod
    def random(domain: "Domain", cliques: List) -> "CliqueVector":
        """Creates a CliqueVector with randomly distributed Factors."""
        return CliqueVector(
            {cl: Factor.random(domain.project(cl)) for cl in cliques}
        )

    @staticmethod
    def from_data(data: "Dataset", cliques: List) -> "CliqueVector":
        """Creates a CliqueVector from data."""
        ans = {}
        for cl in cliques:
            mu = data.project(cl)
            ans[cl] = Factor(mu.domain, mu.datavector())
        return CliqueVector(ans)

    def combine(self, other: "CliqueVector") -> None:
        """Combines this CliqueVector with another."""
        for cl in other:
            if cl in self:
                self[cl] += other[cl]

    def __mul__(self, const: float) -> "CliqueVector":
        return CliqueVector({cl: const * self[cl] for cl in self})

    def __rmul__(self, const: float) -> "CliqueVector":
        return self * const

    def __add__(self, other) -> "CliqueVector":
        if np.isscalar(other):
            return CliqueVector({cl: self[cl] + other for cl in self})
        return CliqueVector({cl: self[cl] + other[cl] for cl in self})

    def __sub__(self, other) -> "CliqueVector":
        return self + (-1 * other)

    def exp(self) -> "CliqueVector":
        """Applies the exponential function to each Factor."""
        return CliqueVector({cl: self[cl].exp() for cl in self})

    def log(self) -> "CliqueVector":
        """Applies the logarithm function to each Factor."""
        return CliqueVector({cl: self[cl].log() for cl in self})

    def dot(self, other: "CliqueVector") -> float:
        """Computes the dot product with another CliqueVector."""
        return sum((self[cl] * other[cl]).sum() for cl in self)

    def size(self) -> int:
        """Returns the total size of the CliqueVector."""
        return sum(self[cl].domain.size() for cl in self)
