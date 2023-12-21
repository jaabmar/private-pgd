from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from inference.domain import Domain


class Factor:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, domain: "Domain", values: torch.tensor):
        """
        Initialize a factor over the given domain.

        Args:
            domain (Domain): The domain of the dataset.
            values (torch.tensor): The array of factor values for each element of the domain.
                                 Values may be a flattened 1D array or an ndarray with the same shape as the domain.

        Raises:
            AssertionError: If the domain size does not match the size of the values or if the shape of values is invalid.
        """
        if isinstance(values, np.ndarray):
            values = torch.tensor(
                values, dtype=torch.float32, device=Factor.device
            )
        assert (
            domain.size() == values.nelement()
        ), "Domain size does not match values size"
        assert (
            len(values.shape) == 1 or values.shape == domain.shape
        ), "Invalid shape for values array"
        self.domain = domain
        self.values = values.reshape(domain.shape).to(Factor.device)

    @staticmethod
    def zeros(domain: "Domain") -> "Factor":
        """
        Create a Factor with all zeros over the given domain.

        Args:
            domain (Domain): The domain of the factor.

        Returns:
            Factor: A Factor object with all values set to zero.
        """
        return Factor(domain, torch.zeros(domain.shape, device=Factor.device))

    @staticmethod
    def ones(domain: "Domain") -> "Factor":
        """
        Create a Factor with all ones over the given domain.

        Args:
            domain (Domain): The domain of the factor.

        Returns:
            Factor: A Factor object with all values set to one.
        """
        return Factor(domain, torch.ones(domain.shape, device=Factor.device))

    @staticmethod
    def random(domain: "Domain") -> "Factor":
        """
        Create a Factor with random values over the given domain.

        Args:
            domain (Domain): The domain of the factor.

        Returns:
            Factor: A Factor object with random values.
        """
        return Factor(domain, torch.rand(domain.shape, device=Factor.device))

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
        vals = torch.zeros(domain.shape, device=Factor.device)
        vals[idx] = -np.inf
        return Factor(domain, vals)

    def expand(self, domain):
        assert domain.contains(
            self.domain
        ), "expanded domain must contain current domain"
        dims = len(domain) - len(self.domain)
        values = self.values.view(self.values.size() + tuple([1] * dims))
        ax = domain.axes(self.domain.attrs)
        # need to find replacement for moveaxis
        ax = ax + tuple(i for i in range(len(domain)) if i not in ax)
        ax = tuple(np.argsort(ax))
        values = values.permute(ax)
        values = values.expand(domain.shape)
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
        ), "attrs must be same as domain attributes"
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        ax = tuple(np.argsort(ax))
        values = self.values.permute(ax)
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
        assert agg in ["sum", "logsumexp"], "agg must be sum or logsumexp"
        marginalized = self.domain.marginalize(attrs)
        if agg == "sum":
            ans = self.sum(marginalized.attrs)
        elif agg == "logsumexp":
            ans = self.logsumexp(marginalized.attrs)
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
            return float(self.values.sum())
        elif attrs == tuple():
            return self
        axes = self.domain.axes(attrs)
        values = self.values.sum(dim=axes)
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
            return float(
                self.values.logsumexp(dim=tuple(range(len(self.values.shape))))
            )
        elif attrs == tuple():
            return self
        axes = self.domain.axes(attrs)
        values = self.values.logsumexp(dim=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

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
            return float(self.values.max())
        return NotImplementedError  # torch.max does not behave like numpy

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
            return Factor(self.domain, self.values.clone())
        np.copyto(out.values, self.values)
        return out

    def __mul__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, other * self.values)
        # print(self.values.max(), other.values.max(), self.domain, other.domain)
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
        zero = torch.tensor(0.0, device=Factor.device)
        inf = torch.tensor(np.inf, device=Factor.device)
        values = torch.where(other.values == -inf, zero, -other.values)
        other = Factor(other.domain, values)
        return self + other

    def __truediv__(self, other):
        # assert np.isscalar(other), 'divisor must be a scalar'
        if np.isscalar(other):
            return self * (1.0 / other)
        tmp = other.expand(self.domain)
        vals = torch.div(self.values, tmp.values)
        vals[tmp.values <= 0] = 0.0
        return Factor(self.domain, vals)

    def exp(self, out: Optional["Factor"] = None) -> "Factor":
        """
        Compute the exponential of the factor's values.

        Args:
            out (Optional[Factor]): The factor to store the result in. If None, creates a new factor.

        Returns:
            Factor: The factor with exponential values.
        """
        if out is None:
            return Factor(self.domain, self.values.exp())
        torch.exp(self.values, out=out.values)
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
            return Factor(self.domain, torch.log(self.values + 1e-100))
        torch.log(self.values, out=out.values)
        return out

    def datavector(self, flatten: bool = True) -> np.ndarray:
        """
        Materialize the factor's values into a data vector.

        Args:
            flatten (bool): If True, flattens the values into a 1D array. Defaults to True.

        Returns:
            np.ndarray: The data vector.
        """
        ans = self.values.detach().to("cpu").numpy()
        return ans.flatten() if flatten else ans

    def torch_datavector(self, flatten=True):
        """Materialize the data vector as a numpy array
        Args:
            flatten (bool): If True, flattens the values into a 1D array. Defaults to True.

        Returns:
            np.ndarray: The data vector.
        """
        ans = self.values
        return ans.flatten() if flatten else ans
