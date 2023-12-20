import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from inference.pgm.inference import FactoredInference
    from inference.privpgd.inference import AdvancedSlicedInference
    from inference.torch_factor import Factor


class CallBack:
    """
    A CallBack is a function called after every iteration of an iterative optimization procedure.
    It is useful for tracking loss and other metrics over time.
    """

    def __init__(
        self,
        engine: Union["FactoredInference", "AdvancedSlicedInference"],
        frequency: int = 50,
    ):
        """
        Initialize the callback object.

        Args:
            engine (Union[FactoredInference, AdvancedSlicedInference]): The FactoredInference object performing the
            optimization.
            frequency (int): The number of iterations to perform before computing the callback function.
        """
        self.engine = engine
        self.frequency = frequency
        self.calls = 0
        self.start = time.time()

    def run(self, marginals: Dict[Tuple[str, ...], "Factor"]) -> None:
        """
        The method to run on each callback. Override this method when creating a subclass.

        Args:
            marginals (Dict[Tuple[str, ...], Factor]): The current marginals at this iteration of the optimization.
        """

    def __call__(self, marginals: Dict[Tuple[str, ...], "Factor"]) -> None:
        """
        Makes the object callable. Invoked in each iteration of the optimization process.

        Args:
            marginals (Dict[Tuple[str, ...], Factor]): The current marginals at this iteration of the optimization.
        """
        if self.calls % self.frequency == 0:
            self.run(marginals)
        self.calls += 1


class Logger(CallBack):
    """
    Logger is the default callback function. It tracks the time, L1 loss, L2 loss, and
    optionally the total variation distance to the true query answers (when available).
    The last is for debugging purposes only - in practice, the true answers cannot be observed.
    """

    def __init__(
        self,
        engine: Union["FactoredInference", "AdvancedSlicedInference"],
        true_answers: Optional[Dict[str, np.ndarray]] = None,
        frequency: int = 50,
    ):
        """
        Initialize the Logger callback object.

        Args:
            engine (Union[FactoredInference, AdvancedSlicedInference]): The FactoredInference object performing the
            optimization.
            true_answers (Optional[Dict[str, np.ndarray]]): A dictionary containing true answers to measurement queries.
            frequency (int): The number of iterations to perform before computing the callback function.
        """
        super().__init__(engine, frequency)
        self.true_answers = true_answers
        self.idx = 0
        self.results = pd.DataFrame()

    def setup(self) -> None:
        """
        Setup method for initializing logging parameters and printing initial information.
        """
        model = self.engine.model
        total = sum(model.domain.size(cl) for cl in model.cliques)
        print("Total clique size:", total, flush=True)
        # cl = max(model.cliques, key=lambda cl: model.domain.size(cl))
        # print('Maximal clique', cl, model.domain.size(cl), flush=True)
        cols = ["iteration", "time", "l1_loss", "l2_loss", "feasibility"]
        if self.true_answers is not None:
            cols.append("variation")
        self.results = pd.DataFrame(columns=cols)
        print("\t\t".join(cols), flush=True)

    def variational_distances(
        self, marginals: Dict[Tuple[str, ...], "Factor"]
    ) -> List[float]:
        """
        Calculate variational distances for the given marginals.

        Args:
            marginals (Dict[Tuple[str, ...], Factor]): Current marginals of the optimization.

        Returns:
            List[float]: A list of variational distances.
        """
        errors = []
        for Q, y, proj in self.true_answers:
            for cl in marginals:
                if set(proj) <= set(cl):
                    mu = marginals[cl].project(proj)
                    x = mu.values.flatten()
                    diff = Q.dot(x) - y
                    err = 0.5 * np.abs(diff).sum() / y.sum()
                    errors.append(err)
                    break
        return errors

    def primal_feasibility(self, mu: Dict[Tuple[str, ...], "Factor"]) -> float:
        """
        Calculate the primal feasibility of the current marginals.

        Args:
            mu (Dict[Tuple[str, ...], Factor]): Current marginals of the optimization.

        Returns:
            float: The primal feasibility value.
        """
        ans = 0
        count = 0
        for r in mu:
            for s in mu:
                if r == s:
                    break
                d = tuple(set(r) & set(s))
                if len(d) > 0:
                    x = mu[r].project(d).datavector()
                    y = mu[s].project(d).datavector()
                    err = np.linalg.norm(x - y, 1)
                    ans += err
                    count += 1
        try:
            return ans / count
        except ZeroDivisionError:
            return 0

    def run(self, marginals: Dict[Tuple[str, ...], "Factor"]) -> None:
        """
        The method run on each callback iteration. It calculates and logs various metrics.

        Args:
            marginals (Dict[Tuple[str, ...], Factor]): The current marginals at this iteration of the optimization.
        """
        if self.idx == 0:
            self.setup()

        t = time.time() - self.start
        l1_loss = self.engine._marginal_loss(marginals, metric="L1")[0]
        l2_loss = self.engine._marginal_loss(marginals, metric="L2")[0]
        feasibility = self.primal_feasibility(marginals)
        row = [self.calls, t, l1_loss, l2_loss, feasibility]
        if self.true_answers is not None:
            variational = np.mean(self.variational_distances(marginals))
            row.append(100 * variational)
        self.results.loc[self.idx] = row
        self.idx += 1

        print("\t\t".join(["%.2f" % v for v in row]), flush=True)
