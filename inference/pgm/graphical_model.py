import itertools
import pickle
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from inference.clique_vector import CliqueVector
from inference.dataset import Dataset
from inference.domain import Domain
from inference.pgm.junction_tree import JunctionTree
from inference.pgm.utils_pgm import (
    greedy_order,
    variable_elimination,
    variable_elimination_logspace,
)
from inference.torch_factor import Factor


class GraphicalModel:
    def __init__(
        self,
        domain: "Domain",
        cliques: List[Tuple[str, ...]],
        total: float = 1.0,
        elimination_order: Optional[Union[int, List[str]]] = None,
    ):
        """
        Initializes a GraphicalModel which is a representation of a probabilistic graphical model.

        Args:
            domain (Domain): The domain of the dataset.
            cliques (List[Tuple[str, ...]]): A list of cliques, each represented as a tuple or list of attributes.
            total (float): The normalization constant for the distribution.
            elimination_order (Optional[Union[int, List[str]]]): An elimination order for the JunctionTree algorithm.
                                                                 An integer indicates the number of stochastic trials for
                                                                 elimination order determination.
                                                                 Defaults to None, using a greedy elimination order.
        """
        self.domain = domain
        self.total = total
        self.junction_tree = JunctionTree(domain, cliques, elimination_order)

        self.cliques = self.junction_tree.maximal_cliques()
        self.message_order = self.junction_tree.mp_order()
        self.sep_axes = self.junction_tree.separator_axes()
        self.neighbors = self.junction_tree.neighbors()
        self.elimination_order = self.junction_tree.elimination_order
        self.marginals = None
        self.potentials = None

        self.size = sum(domain.size(cl) for cl in self.cliques)
        if self.size * 8 > 4 * 10**9:
            import warnings

            message = (
                f"Size of parameter vector is {self.size * 8 / 10**9:.2f} GB. "
            )
            message += "Consider removing some measurements or finding a better elimination order."
            warnings.warn(message)

    @staticmethod
    def save(model: "GraphicalModel", path: str) -> None:
        """
        Saves the GraphicalModel to a file.

        Args:
            model (GraphicalModel): The GraphicalModel instance to save.
            path (str): The file path to save the model to.
        """
        pickle.dump(model, open(path, "wb"))

    @staticmethod
    def load(path: str) -> "GraphicalModel":
        """
        Loads a GraphicalModel from a file.

        Args:
            path (str): The file path from which to load the model.

        Returns:
            GraphicalModel: The loaded GraphicalModel instance.
        """
        return pickle.load(open(path, "rb"))

    def project(self, attrs: Union[List[str], Tuple[str, ...]]) -> "Factor":
        """
        Projects the distribution onto a subset of attributes, computing the marginal distribution.

        Args:
            attrs (Union[List[str], Tuple[str, ...]]): A subset of attributes in the domain to project onto.

        Returns:
            Factor: A Factor object representing the marginal distribution.
        """
        if isinstance(attrs, list):
            attrs = tuple(attrs)
        if self.marginals:
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)

        elim_order = greedy_order(
            self.domain, self.cliques + [attrs], self.domain.invert(attrs)
        )
        pots = list(self.potentials.values())
        ans = variable_elimination_logspace(pots, elim_order, self.total)
        return ans.project(attrs)

    def krondot(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Compute the Kronecker product dot product for a set of query matrices on each attribute.

        Args:
            matrices (List[np.ndarray]): A list of query matrices for each attribute in the domain.

        Returns:
            np.ndarray: The resulting vector of query answers.
        """
        assert all(
            M.shape[1] == n for M, n in zip(matrices, self.domain.shape)
        ), "Matrices must conform to the shape of the domain."
        logZ = self.belief_propagation(self.potentials, logZ=True)
        factors = [self.potentials[cl].exp() for cl in self.cliques]
        factor = type(factors[0])  # Infer the type of the factors
        elim = self.domain.attrs
        for attr, Q in zip(elim, matrices):
            d = Domain(["%s-answer" % attr, attr], Q.shape)
            factors.append(factor(d, Q))
        result = variable_elimination(factors, elim)
        result = result.transpose(["%s-answer" % a for a in elim])
        return result.datavector(flatten=False) * self.total / np.exp(logZ)

    def calculate_many_marginals(
        self, projections: List[Union[List[str], Tuple[str, ...]]]
    ) -> Dict[Union[List[str], Tuple[str, ...]], "Factor"]:
        """
        Calculates marginals for multiple projections using an efficient algorithm for out-of-clique queries.

        Args:
            projections (List[Union[List[str], Tuple[str, ...]]]): A list of projections, each a subset of attributes.

        Returns:
            Dict[Union[List[str], Tuple[str, ...]], Factor]: A dictionary of marginals, each represented as a Factor.
        """

        self.marginals = self.belief_propagation(self.potentials)
        sep = self.sep_axes
        neighbors = self.neighbors
        # first calculate P(Cj | Ci) for all neighbors Ci, Cj
        conditional = {}
        for Ci in neighbors:
            for Cj in neighbors[Ci]:
                Sij = sep[(Cj, Ci)]
                Z = self.marginals[Cj]
                conditional[(Cj, Ci)] = Z / Z.project(Sij)

        # now iterate through pairs of cliques in order of distance
        pred, dist = nx.floyd_warshall_predecessor_and_distance(
            self.junction_tree.tree, weight=False
        )
        results = {}
        for Ci, Cj in sorted(
            itertools.combinations(self.cliques, 2),
            key=lambda X: dist[X[0]][X[1]],
        ):
            Cl = pred[Ci][Cj]
            Y = conditional[(Cj, Cl)]
            if Cl == Ci:
                X = self.marginals[Ci]
                results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
            else:
                X = results[(Ci, Cl)]
                S = set(Cl) - set(Ci) - set(Cj)
                results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(S)

        results = {
            self.domain.canonical(key[0] + key[1]): results[key]
            for key in results
        }

        answers = {}
        for proj in projections:
            for attr in results:
                if set(proj) <= set(attr):
                    answers[proj] = results[attr].project(proj)
                    break
            if proj not in answers:
                # just use variable elimination
                answers[proj] = self.project(proj)

        return answers

    def datavector(self, flatten: bool = True) -> np.ndarray:
        """
        Materializes the explicit representation of the distribution as a data vector.

        Args:
            flatten (bool): If True, flattens the resulting data vector. Defaults to True.

        Returns:
            np.ndarray: The data vector representing the distribution.
        """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def belief_propagation(
        self, potentials: Dict[Tuple[str, ...], "Factor"], logZ: bool = False
    ) -> Union["CliqueVector", float]:
        """
        Computes the marginals of the graphical model using belief propagation.

        Args:
            potentials (Dict[Tuple[str, ...], Factor]): The (log-space) parameters of the graphical model.
            logZ (bool): If True, returns the log partition function instead of marginals. Defaults to False.

        Returns:
            Union[CliqueVector, float]: The marginals of the graphical model if logZ is False, else the log partition.
        """
        beliefs = {cl: potentials[cl].copy() for cl in potentials}
        messages = {}
        for i, j in self.message_order:
            sep = beliefs[i].domain.invert(self.sep_axes[(i, j)])
            if (j, i) in messages:
                tau = beliefs[i] - messages[(j, i)]
            else:
                tau = beliefs[i]
            messages[(i, j)] = tau.logsumexp(sep)
            beliefs[j] += messages[(i, j)]

        cl = self.cliques[0]
        if logZ:
            return beliefs[cl].logsumexp()

        logZ = beliefs[cl].logsumexp()
        for cl in self.cliques:
            beliefs[cl] += np.log(self.total) - logZ
            beliefs[cl] = beliefs[cl].exp(out=beliefs[cl])
        return CliqueVector(beliefs)

    def mle(self, marginals: Dict[Tuple[str, ...], "Factor"]) -> "CliqueVector":
        """
        Computes the model parameters from the given marginals.

        Args:
            marginals (Dict[Tuple[str, ...], Factor]): The target marginals of the distribution.

        Returns:
            CliqueVector: The potentials of the graphical model with the given marginals.
        """
        potentials = {}
        variables = set()
        for cl in self.cliques:
            new = tuple(variables & set(cl))
            # factor = marginals[cl] / marginals[cl].project(new)
            variables.update(cl)
            potentials[cl] = (
                marginals[cl].log() - marginals[cl].project(new).log()
            )
        return CliqueVector(potentials)

    def fit(self, data: "Dataset") -> None:
        """
        Fits the graphical model to the given data.

        Args:
            data (Dataset): The dataset to fit the model to.

        Raises:
            AssertionError: If the model domain is not compatible with the data domain.
        """
        assert data.domain.contains(
            self.domain
        ), "model domain not compatible with data domain"
        marginals = {}
        for cl in self.cliques:
            x = data.project(cl).datavector()
            dom = self.domain.project(cl)
            marginals[cl] = Factor(dom, x)
        self.potentials = self.mle(marginals)

    def synthetic_data(
        self, rows: Optional[int] = None, method: str = "round"
    ) -> "Dataset":
        """
        Generates synthetic tabular data from the distribution.

        Args:
            rows (Optional[int]): The number of rows in the synthetic dataset. If None, uses the total count of the model.
                                Defaults to None.
            method (str): The method to generate data, either 'round' or 'sample'. Defaults to 'round'.

        Returns:
            Dataset: The generated synthetic dataset.
        """
        total = int(self.total) if rows is None else rows
        cols = self.domain.attrs
        data = np.zeros((total, len(cols)), dtype=int)
        df = pd.DataFrame(data, columns=cols)
        cliques = [set(cl) for cl in self.cliques]

        def synthetic_col(counts, total):
            if method == "sample":
                probas = counts / counts.sum()
                return np.random.choice(counts.size, total, True, probas)
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)
            integ = integ.astype(int)
            extra = total - integ.sum()
            if extra > 0:
                idx = np.random.choice(
                    counts.size, extra, False, frac / frac.sum()
                )
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = self.elimination_order[::-1]
        col = order[0]
        marg = self.project([col]).datavector(flatten=False)
        df.loc[:, col] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set.union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            marg = self.project(proj + (col,)).datavector(flatten=False)

            def foo(group):
                idx = group.name
                vals = synthetic_col(marg[idx], group.shape[0])
                group[col] = vals
                return group

            if len(proj) >= 1:
                df = df.groupby(list(proj), group_keys=False).apply(foo)
            else:
                df[col] = synthetic_col(marg, df.shape[0])

        return Dataset(df, self.domain)
