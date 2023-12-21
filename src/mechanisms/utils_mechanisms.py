import itertools
from typing import List, Optional

import numpy as np

from inference.dataset import Dataset


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def generate_all_kway_workload(
    data: "Dataset",
    degree: int = 2,
    num_marginals: Optional[int] = None,
) -> List[tuple]:
    """
    Generate all k-way marginals workload.

    Args:
        data (Dataset): The dataset to generate workload for.
        degree (int): The degree of combinations.
        num_marginals (Optional[int]): Number of marginals to generate.

    Returns:
        List[tuple]: A list of attribute combinations.
    """
    workload = list(itertools.combinations(data.domain.attrs, degree))
    if num_marginals is not None:
        workload = [
            workload[i]
            for i in np.random.choice(
                len(workload), num_marginals, replace=False
            )
        ]
    return workload
