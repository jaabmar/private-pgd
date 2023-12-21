import itertools
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from inference.domain import Domain


class JunctionTree:
    """
    A JunctionTree is a tree structure representing a transformation of a GraphicalModel.
    It identifies the maximal cliques in the graphical model and specifies the message passing
    order for belief propagation. The tree is characterized by an elimination order,
    which is chosen based on a greedy algorithm or can be explicitly provided.

    Attributes:
        cliques (List[Tuple[str, ...]]): Maximal cliques in the graphical model.
        domain (Domain): The domain of the dataset.
        graph (networkx.Graph): The underlying graph representing the junction tree.
        tree (networkx.Graph): The actual junction tree structure.
        order (List[str]): The elimination order used to construct the tree.
    """

    def __init__(
        self,
        domain: "Domain",
        cliques: List[Tuple[str, ...]],
        elimination_order: Optional[Union[int, List[str]]] = None,
    ):
        """
        Initializes the JunctionTree object.

        Args:
            domain (Domain): The domain associated with the graphical model.
            cliques (List[Tuple[str, ...]]): List of tuples representing the cliques in the graphical model.
            elimination_order (Optional[Union[int, List[str]]]): The elimination order for constructing the tree.
                If an integer is provided, it denotes the number of stochastic trials for order determination.
                Defaults to None, which implies a greedy determination of order.
        """
        self.cliques = [tuple(cl) for cl in cliques]
        self.domain = domain
        self.graph = self._make_graph()
        self.tree, self.order = self._make_tree(elimination_order)

    def maximal_cliques(self) -> List[Tuple[str, ...]]:
        """
        Returns the list of maximal cliques in the model.

        Returns:
            List[Tuple[str, ...]]: A list of tuples, each representing a maximal clique.
        """
        return list(nx.dfs_preorder_nodes(self.tree))

    def mp_order(self) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """
        Determines a valid message passing order.

        Returns:
            List[Tuple[Tuple[str, ...], Tuple[str, ...]]]: A list of tuples representing the message passing order.
        """
        edges = set()
        messages = [(a, b) for a, b in self.tree.edges()] + [
            (b, a) for a, b in self.tree.edges()
        ]
        for m1 in messages:
            for m2 in messages:
                if m1[1] == m2[0] and m1[0] != m2[1]:
                    edges.add((m1, m2))
        G = nx.DiGraph()
        G.add_nodes_from(messages)
        G.add_edges_from(edges)
        return list(nx.topological_sort(G))

    def separator_axes(
        self,
    ) -> Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[str, ...]]:
        """
        Computes the separator axes between cliques in the message passing order.

        Returns:
            Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[str, ...]]: A dictionary mapping pairs of cliques.
        """
        return {(i, j): tuple(set(i) & set(j)) for i, j in self.mp_order()}

    def neighbors(self) -> Dict[Tuple[str, ...], Set[Tuple[str, ...]]]:
        """
        Finds the neighbors of each clique in the junction tree.

        Returns:
            Dict[Tuple[str, ...], Set[Tuple[str, ...]]]: A dictionary mapping each clique to a set of its neighbors.
        """
        return {i: set(self.tree.neighbors(i)) for i in self.maximal_cliques()}

    def _make_graph(self) -> nx.Graph:
        """
        Constructs the underlying graph from the cliques.

        Returns:
            nx.Graph: The constructed graph.
        """
        G = nx.Graph()
        G.add_nodes_from(self.domain.attrs)
        for cl in self.cliques:
            G.add_edges_from(itertools.combinations(cl, 2))
        return G

    def _triangulated(self, order: List[str]) -> Tuple[nx.Graph, int]:
        """
        Constructs a triangulated graph based on the given elimination order.

        Args:
            order (List[str]): The elimination order.

        Returns:
            Tuple[nx.Graph, int]: The triangulated graph and its associated cost.
        """
        edges = set()
        G = nx.Graph(self.graph)
        for node in order:
            tmp = set(itertools.combinations(G.neighbors(node), 2))
            edges |= tmp
            G.add_edges_from(tmp)
            G.remove_node(node)
        tri = nx.Graph(self.graph)
        tri.add_edges_from(edges)
        cliques = [tuple(c) for c in nx.find_cliques(tri)]
        cost = sum(self.domain.project(cl).size() for cl in cliques)
        return tri, cost

    def _greedy_order(self, stochastic: bool = True) -> Tuple[List[str], int]:
        """
        Generates an elimination order using a greedy algorithm.

        Args:
            stochastic (bool): If True, uses a stochastic approach for generating the order.

        Returns:
            Tuple[List[str], int]: The generated elimination order and its associated cost.
        """

        order = []
        domain, cliques = self.domain, self.cliques
        unmarked = list(domain.attrs)
        cliques = set(cliques)
        total_cost = 0
        for _ in range(len(domain)):
            cost = OrderedDict()
            for a in unmarked:
                # all cliques that have a
                neighbors = list(filter(lambda cl: a in cl, cliques))
                # variables in this "super-clique"
                variables = tuple(set.union(set(), *map(set, neighbors)))
                # domain for the resulting factor
                newdom = domain.project(variables)
                # cost of removing a
                cost[a] = newdom.size()

            # find the best variable to eliminate
            if stochastic:
                choices = list(unmarked)
                costs = np.array([cost[a] for a in choices], dtype=float)
                probas = np.max(costs) - costs + 1
                probas /= probas.sum()
                i = np.random.choice(probas.size, p=probas)
                a = choices[i]
                # print(choices, probas)
            else:
                a = min(cost, key=lambda a: cost[a])

            # do some cleanup
            order.append(a)
            unmarked.remove(a)
            neighbors = list(filter(lambda cl: a in cl, cliques))
            variables = tuple(set.union(set(), *map(set, neighbors)) - {a})
            cliques -= set(neighbors)
            cliques.add(variables)
            total_cost += cost[a]

        return order, total_cost

    def _make_tree(
        self, order: Optional[Union[int, List[str]]] = None
    ) -> Tuple[nx.Graph, List[str]]:
        """
        Constructs the junction tree based on the given or computed elimination order.

        Args:
            order (Optional[Union[int, List[str]]]): The elimination order. If an integer is provided, it denotes the number
                                                    of stochastic trials. Defaults to None, which uses a deterministic
                                                    greedy approach.

        Returns:
            Tuple[nx.Graph, List[str]]: The constructed junction tree and the elimination order used.
        """
        if order is None:
            order = self._greedy_order(stochastic=False)[0]
        elif type(order) is int:
            orders = [self._greedy_order(stochastic=False)] + [
                self._greedy_order(stochastic=True) for _ in range(order)
            ]
            order = min(orders, key=lambda x: x[1])[0]
        self.elimination_order = order
        tri, _ = self._triangulated(order)
        cliques = sorted(
            [self.domain.canonical(c) for c in nx.find_cliques(tri)]
        )
        complete = nx.Graph()
        complete.add_nodes_from(cliques)
        for c1, c2 in itertools.combinations(cliques, 2):
            wgt = len(set(c1) & set(c2))
            complete.add_edge(c1, c2, weight=-wgt)
        spanning = nx.minimum_spanning_tree(complete)
        return spanning, order
