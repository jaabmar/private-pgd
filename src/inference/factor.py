from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from scipy.special import logsumexp

if TYPE_CHECKING:
    from inference.domain import Domain


class Factor:
    def __init__(self, domain: "Domain", values: np.ndarray):
        """
        Initialize a factor over the given domain.

        Args:
            domain (Domain): The domain of the dataset.
            values (np.ndarray): The array of factor values for each element of the domain.
                                 Values may be a flattened 1D array or an ndarray with the same shape as the domain.

        Raises:
            AssertionError: If the domain size does not match the size of the values or if the shape of values is invalid.
        """
        assert (
            domain.size() == values.size
        ), "Domain size does not match values size."
        assert (
            values.ndim == 1 or values.shape == domain.shape
        ), "Invalid shape for values array."
        self.domain = domain
        self.values = values.reshape(domain.shape)

    @staticmethod
    def zeros(domain: "Domain") -> "Factor":
        """
        Create a Factor with all zeros over the given domain.

        Args:
            domain (Domain): The domain of the factor.

        Returns:
            Factor: A Factor object with all values set to zero.
        """
        return Factor(domain, np.zeros(domain.shape))

    @staticmethod
    def ones(domain: "Domain") -> "Factor":
        """
        Create a Factor with all ones over the given domain.

        Args:
            domain (Domain): The domain of the factor.

        Returns:
            Factor: A Factor object with all values set to one.
        """
        return Factor(domain, np.ones(domain.shape))

    @staticmethod
    def random(domain: "Domain") -> "Factor":
        """
        Create a Factor with random values over the given domain.

        Args:
            domain (Domain): The domain of the factor.

        Returns:
            Factor: A Factor object with random values.
        """
        return Factor(domain, np.random.rand(*domain.shape))

    @staticmethod
    def uniform(domain: "Domain") -> "Factor":
        """
        Create a uniform Factor over the given domain.

        Args:
            domain (Domain): The domain of the factor.

        Returns:
            Factor: A Factor object with uniform values.
        """
        return Factor.ones(domain) / domain.size()

    @staticmethod
    def active(
        domain: "Domain", structural_zeros: List[Union[int, float]]
    ) -> "Factor":
        """
        Create a Factor that is 0 everywhere except in positions present in 'structural_zeros', where it is -infinity.

        Args:
            domain (Domain): The domain of the factor.
            structural_zeros (List[Union[int, float]]): A list of values that are not possible.

        Returns:
            Factor: A Factor object with specified structural zeros.
        """
        idx = tuple(np.array(structural_zeros).T)
        vals = np.zeros(domain.shape)
        vals[idx] = -np.inf
        return Factor(domain, vals)

    def expand(self, domain: "Domain") -> "Factor":
        """
        Expand the factor to a larger domain.

        Args:
            domain (Domain): The new domain to expand to.

        Returns:
            Factor: A new Factor object expanded to the specified domain.

        Raises:
            AssertionError: If the new domain does not contain the current domain.
        """
        assert domain.contains(
            self.domain
        ), "Expanded domain must contain current domain."
        dims = len(domain) - len(self.domain)
        values = self.values.reshape(self.domain.shape + tuple([1] * dims))
        ax = domain.axes(self.domain.attrs)
        values = np.moveaxis(values, range(len(ax)), ax)
        values = np.broadcast_to(values, domain.shape)
        return Factor(domain, values)

    def transpose(self, attrs: List[str]) -> "Factor":
        """
        Transpose the factor to a new order of attributes.

        Args:
            attrs (List[str]): List of attributes to transpose to.

        Returns:
            Factor: A new Factor object with attributes transposed as specified.

        Raises:
            AssertionError: If the attributes do not match the domain attributes.
        """
        assert set(attrs) == set(
            self.domain.attrs
        ), "Attrs must be the same as domain attributes."
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        values = np.moveaxis(self.values, range(len(ax)), ax)
        return Factor(newdom, values)

    def project(self, attrs: List[str], agg: str = "sum") -> "Factor":
        """
        Project the factor onto a list of attributes using aggregation.

        Args:
            attrs (List[str]): List of attributes to project onto.
            agg (str): Aggregation method, either 'sum' or 'logsumexp'.

        Returns:
            Factor: A new Factor object after projection.

        Raises:
            AssertionError: If the aggregation method is not 'sum' or 'logsumexp'.
        """
        assert agg in [
            "sum",
            "logsumexp",
        ], "Aggregation must be 'sum' or 'logsumexp'."
        marginalized = self.domain.marginalize(attrs)
        ans = (
            self.sum(marginalized.attrs)
            if agg == "sum"
            else self.logsumexp(marginalized.attrs)
        )
        return ans.transpose(attrs)

    def sum(self, attrs: Optional[List[str]] = None) -> Union["Factor", float]:
        """
        Sum over the specified attributes or the entire factor.

        Args:
            attrs (Optional[List[str]]): Attributes to sum over. If None, sum over all.

        Returns:
            Union[Factor, float]: Summed factor or a single value if all attributes are summed over.
        """
        if attrs is None:
            return np.sum(self.values)
        axes = self.domain.axes(attrs)
        values = np.sum(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logsumexp(
        self, attrs: Optional[List[str]] = None
    ) -> Union["Factor", float]:
        """
        Apply log-sum-exp over the specified attributes or the entire factor.

        Args:
            attrs (Optional[List[str]]): Attributes to apply log-sum-exp over. If None, apply over all.

        Returns:
            Union[Factor, float]: Factor after log-sum-exp or a single value if all attributes are aggregated.
        """
        if attrs is None:
            return logsumexp(self.values)
        axes = self.domain.axes(attrs)
        values = logsumexp(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logaddexp(self, other: "Factor") -> "Factor":
        """
        Compute the element-wise logarithm of the exponential sum of two factors.

        Args:
            other (Factor): Another factor to perform the operation with.

        Returns:
            Factor: A new factor with the result of the logaddexp operation.
        """
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, np.logaddexp(factor1.values, factor2.values))

    def max(self, attrs: Optional[Union[str, list]] = None) -> "Factor":
        """
        Compute the maximum over given attributes.

        Args:
            attrs (Optional[Union[str, list]]): Attributes over which to compute the maximum.
                                                If None, computes the maximum over all attributes.

        Returns:
            Factor: A new factor with the maximum values.
        """
        if attrs is None:
            return self.values.max()
        axes = self.domain.axes(attrs)
        values = np.max(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def condition(self, evidence: Dict[str, Union[int, slice]]) -> "Factor":
        """
        Condition the factor on given evidence.

        Args:
            evidence (Dict[str, Union[int, slice]]): A dictionary where keys are attributes,
                                                     and values are elements or slices of the domain.

        Returns:
            Factor: A new factor conditioned on the given evidence.
        """
        slices = [
            evidence[a] if a in evidence else slice(None) for a in self.domain
        ]
        newdom = self.domain.marginalize(evidence.keys())
        values = self.values[tuple(slices)]
        return Factor(newdom, values)

    def copy(self, out: Optional["Factor"] = None) -> "Factor":
        """
        Copy the current factor to a new factor or an existing one.

        Args:
            out (Optional[Factor]): The factor to copy the values into. If None, creates a new factor.

        Returns:
            Factor: The copied factor.
        """
        if out is None:
            return Factor(self.domain, self.values.copy())
        np.copyto(out.values, self.values)
        return out

    def exp(self, out: Optional["Factor"] = None) -> "Factor":
        """
        Compute the exponential of the factor's values.

        Args:
            out (Optional[Factor]): The factor to store the result in. If None, creates a new factor.

        Returns:
            Factor: The factor with exponential values.
        """
        if out is None:
            return Factor(self.domain, np.exp(self.values))
        np.exp(self.values, out=out.values)
        return out

    def log(self, out: Optional["Factor"] = None) -> "Factor":
        """
        Compute the natural logarithm of the factor's values, adding a small value for numerical stability.

        Args:
            out (Optional[Factor]): The factor to store the result in. If None, creates a new factor.

        Returns:
            Factor: The factor with logarithmic values.
        """
        if out is None:
            return Factor(self.domain, np.log(self.values + 1e-100))
        np.log(self.values + 1e-100, out=out.values)
        return out

    def datavector(self, flatten: bool = True) -> np.ndarray:
        """
        Materialize the factor's values into a data vector.

        Args:
            flatten (bool): If True, flattens the values into a 1D array. Defaults to True.

        Returns:
            np.ndarray: The data vector.
        """
        return self.values.flatten() if flatten else self.values

    def __mul__(self, other):
        if np.isscalar(other):
            new_values = np.nan_to_num(other * self.values)
            return Factor(self.domain, new_values)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, factor1.values * factor2.values)

    def __add__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, other + self.values)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, factor1.values + factor2.values)

    def __iadd__(self, other):
        if np.isscalar(other):
            self.values += other
            return self
        factor2 = other.expand(self.domain)
        self.values += factor2.values
        return self

    def __imul__(self, other):
        if np.isscalar(other):
            self.values *= other
            return self
        factor2 = other.expand(self.domain)
        self.values *= factor2.values
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, self.values - other)
        other = Factor(
            other.domain, np.where(other.values == -np.inf, 0, -other.values)
        )
        return self + other

    def __truediv__(self, other):
        if np.isscalar(other):
            new_values = self.values / other
            new_values = np.nan_to_num(new_values)
            return Factor(self.domain, new_values)
        tmp = other.expand(self.domain)
        vals = np.divide(self.values, tmp.values, where=tmp.values > 0)
        vals[tmp.values <= 0] = 0.0
        return Factor(self.domain, vals)
