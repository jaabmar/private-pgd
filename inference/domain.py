from functools import reduce
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist


class Domain:
    def __init__(
        self,
        attrs: Iterable[str],
        shape: Iterable[int],
        d: Union[int, None] = None,
    ):
        """
        Construct a Domain object.

        Args:
            attrs (Iterable[str]): A list or tuple of attribute names.
            shape (Iterable[int]): A list or tuple of domain sizes for each attribute.
            d (int, optional): An optional dimension parameter, default is None.

        Raises:
            AssertionError: If the number of attributes does not match the number of shapes.
        """
        assert len(attrs) == len(shape), "Dimensions must be equal."
        self.attrs = tuple(attrs)
        self.shape = tuple(shape)
        self.config = dict(zip(attrs, shape))
        self.centers = None
        self.d = d
        self.embedding_type = {key: "s" for key in self.attrs}

    @staticmethod
    def fromdict(config: Dict[str, int]) -> "Domain":
        """
        Construct a Domain object from a dictionary of {attribute: size} values.

        Args:
            config (Dict[str, int]): A dictionary where keys are attribute names and values are sizes.

        Returns:
            Domain: A new Domain object constructed from the dictionary.
        """
        return Domain(config.keys(), config.values(), d=None)

    def project(self, attrs: Union[str, List[str]]) -> "Domain":
        """
        Project the domain onto a subset of attributes.

        Args:
            attrs (Union[str, List[str]]): The attributes to project onto.

        Returns:
            Domain: The projected Domain object.
        """
        if isinstance(attrs, str):
            attrs = [attrs]
        shape = tuple(self.config[a] for a in attrs)
        return Domain(attrs, shape, self.d)

    def marginalize(self, attrs: Union[str, List[str]]) -> "Domain":
        """
        Marginalize out some attributes from the domain (opposite of project).

        Args:
            attrs (Union[str, List[str]]): The attributes to marginalize out.

        Returns:
            Domain: The marginalized Domain object.
        """
        proj = [a for a in self.attrs if a not in attrs]
        return self.project(proj)

    def axes(self, attrs: Union[str, List[str]]) -> Tuple[int, ...]:
        """
        Return the axes tuple for the given attributes.

        Args:
            attrs (Union[str, List[str]]): The attributes.

        Returns:
            Tuple[int, ...]: A tuple with the corresponding axes.
        """
        if isinstance(attrs, str):
            attrs = [attrs]
        return tuple(self.attrs.index(a) for a in attrs)

    def transpose(self, attrs: Union[str, List[str]]) -> "Domain":
        """
        Reorder the attributes in the Domain object.

        Args:
            attrs (Union[str, List[str]]): The attributes to reorder.

        Returns:
            Domain: The Domain object with reordered attributes.
        """
        if isinstance(attrs, str):
            attrs = [attrs]
        return self.project(attrs)

    def invert(self, attrs: Union[str, List[str]]) -> List[str]:
        """
        Returns the attributes in the domain not in the list.

        Args:
            attrs (Union[str, List[str]]): The attributes to be excluded.

        Returns:
            List[str]: A list of attributes not in 'attrs'.
        """
        if isinstance(attrs, str):
            attrs = [attrs]
        return [a for a in self.attrs if a not in attrs]

    def merge(self, other: "Domain") -> "Domain":
        """
        Merge this domain object with another.

        Args:
            other (Domain): Another Domain object to merge with.

        Returns:
            Domain: A new Domain object covering the full domain of both merged objects.

        Example:
            >>> D1 = Domain(['a','b'], [10,20])
            >>> D2 = Domain(['b','c'], [20,30])
            >>> D1.merge(D2)
            Domain(['a','b','c'], [10,20,30])
        """
        extra = other.marginalize(self.attrs)
        return Domain(
            self.attrs + extra.attrs, self.shape + extra.shape, self.d
        )

    def contains(self, other: "Domain") -> bool:
        """
        Determine if this domain contains another.

        Args:
            other (Domain): Another Domain object to check containment against.

        Returns:
            bool: True if this domain contains the other domain; False otherwise.
        """
        return set(other.attrs) <= set(self.attrs)

    def size(self, attrs: Union[str, List[str], None] = None) -> int:
        """
        Return the total size of the domain.

        Args:
            attrs (Union[str, List[str], None], optional): Specific attributes to calculate size for.
                                                          If None, calculates size for the entire domain.

        Returns:
            int: The total size of the specified domain or the entire domain.
        """
        if attrs is None:
            return reduce(lambda x, y: x * y, self.shape, 1)
        return self.project(attrs).size()

    def sort(self, how: str = "size") -> "Domain":
        """
        Return a new domain object, sorted by attribute size or attribute name.

        Args:
            how (str): Sorting criterion, either "size" or "name". Defaults to "size".

        Returns:
            Domain: A new Domain object sorted based on the specified criterion.
        """
        if how == "size":
            attrs = sorted(self.attrs, key=lambda attr: self.config[attr])
        elif how == "name":
            attrs = sorted(self.attrs)
        else:
            raise ValueError(
                "Invalid sorting criterion. Choose 'size' or 'name'."
            )
        return self.project(attrs)

    def canonical(self, attrs: Union[str, List[str]]) -> Tuple[str, ...]:
        """
        Return the canonical ordering of the attributes.

        Args:
            attrs (Union[str, List[str]]): The attributes for which to find the canonical ordering.

        Returns:
            Tuple[str, ...]: A tuple containing the canonical ordering of the specified attributes.
        """
        if isinstance(attrs, str):
            attrs = [attrs]
        return tuple(a for a in self.attrs if a in attrs)

    def get_centers(
        self, cliques: Optional[List[Tuple[str]]] = None
    ) -> Dict[Tuple[str], np.ndarray]:
        """
        Calculate the center points for each clique in the domain.

        Args:
            cliques (Optional[List[Tuple[str]]]): A list of cliques (subsets of attributes) for which to calculate centers.
                                                  If None, uses the entire set of attributes.

        Returns:
            Dict[Tuple[str], np.ndarray]: A dictionary where keys are cliques (tuples of attribute names) and
                                          values are arrays of center points for those cliques.
        """
        if not cliques:
            cliques = [self.attrs]

        centers = {}
        for subset in cliques:
            arrays = [np.arange(self.config[i]) for i in subset]
            center_unnormalized = np.array(list(product(*arrays)))
            nbins = np.array([self.config[i] for i in subset])
            centers[subset] = (center_unnormalized * 2 + 1) / (2 * nbins)
        return centers

    def get_distance_matrix(
        self, cliques: Optional[List[Tuple[str]]] = None, p: int = 5
    ) -> Dict[Tuple[str], np.ndarray]:
        """
        Compute the distance matrix for each clique using the Minkowski distance.

        Args:
            cliques (Optional[List[Tuple[str]]]): A list of tuples representing cliques
                                                  for which the distance matrices are computed.
                                                  Defaults to None, which uses all attributes.
            p (int): The order of the Minkowski distance. Defaults to 5.

        Returns:
            Dict[Tuple[str], np.ndarray]: A dictionary where keys are tuples of attributes
                                          (cliques) and values are distance matrices.
        """
        if cliques is None:
            cliques = [self.attrs]

        centers = self.get_centers(cliques)
        mat = {}
        for cl in cliques:
            mat[cl] = cdist(centers[cl], centers[cl], "minkowski", p=p)
        return mat

    def __contains__(self, attr):
        return attr in self.attrs

    def __getitem__(self, a):
        return self.config[a]

    def __iter__(self):
        return self.attrs.__iter__()

    def __len__(self):
        return len(self.attrs)

    def __eq__(self, other):
        return self.attrs == other.attrs and self.shape == other.shape

    def __repr__(self):
        inner = ", ".join(["%s: %d" % x for x in zip(self.attrs, self.shape)])
        return "Domain(%s)" % inner

    def __str__(self):
        return self.__repr__()
