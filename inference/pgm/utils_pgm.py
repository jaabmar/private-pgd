from functools import reduce

import numpy as np


def variable_elimination_logspace(potentials, elim, total):
    """run variable elimination on a list of **logspace** factors"""
    k = len(potentials)
    psi = dict(zip(range(k), potentials))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x, y: x + y, psi2, 0)
        tau = phi.logsumexp([z])
        psi[k] = tau
        k += 1
    ans = reduce(lambda x, y: x + y, psi.values(), 0)
    return (ans - ans.logsumexp() + np.log(total)).exp()


def variable_elimination(factors, elim):
    """run variable elimination on a list of (non-logspace) factors"""
    k = len(factors)
    psi = dict(zip(range(k), factors))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x, y: x * y, psi2, 1)
        tau = phi.sum([z])
        psi[k] = tau
        k += 1
    return reduce(lambda x, y: x * y, psi.values(), 1)


def greedy_order(domain, cliques, elim):
    order = []
    unmarked = set(elim)
    cliques = set(cliques)
    total_cost = 0
    for _ in range(len(elim)):
        cost = {}
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
        a = min(cost, key=lambda a: cost[a])

        # do some cleanup
        order.append(a)
        unmarked.remove(a)
        neighbors = list(filter(lambda cl: a in cl, cliques))
        variables = tuple(set.union(set(), *map(set, neighbors)) - {a})
        cliques -= set(neighbors)
        cliques.add(variables)
        total_cost += cost[a]

    return order
