"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
from abc import ABC, abstractmethod
import numpy as np


class PenaltyConstraintHandler(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, constraints_values, curr_ctr, max_ctr):
        pass

    def __call__(self, constraints_values, curr_ctr=None, max_ctr=None):
        return self.evaluate(constraints_values=constraints_values, curr_ctr=curr_ctr, max_ctr=max_ctr)


class DefaultPenaltyConstraintHandler(PenaltyConstraintHandler):
    def __init__(self) -> None:
        pass

    def evaluate(self, constraints_values, curr_ctr=None, max_ctr=None):
        return 0


class RangeConstraint:
    def __init__(self, factor: float,  low=-np.inf, high=np.inf):
        self.factor = factor
        self.range = np.asarray([low, high])


class EqualityConstraint(RangeConstraint):
    def __init__(self, factor: float, value: float, tolerance: float):
        super().__init__(factor=factor, low=value - tolerance, high=value + tolerance)


class StaticPenaltyConstraintHandler(PenaltyConstraintHandler):

    def __init__(self, constraints: list[RangeConstraint], beta=2) -> None:
        super().__init__()
        self.constraints = constraints
        self.beta = beta
        self.load_constraints(constraints)

    def load_constraints(self, constraints):
        ranges = np.empty((len(constraints), 2))
        factors = np.ones(len(constraints))
        for idx, constraint in enumerate(constraints):
            ranges[idx] = constraint.range
            factors[idx] = constraint.factor
        self.ranges = ranges
        self.factors = factors

    def evaluate(self, constraints_values, **kwargs):
        n_constraints = len(self.constraints)
        constraints_values_np = np.asarray(constraints_values)
        if constraints_values_np.shape[1] != n_constraints:
            raise Exception(
                f"Invalid Shape of constraints_values: Expected (Any, {n_constraints}) but got {constraints_values_np.shape}")
        n_individuals = constraints_values_np.shape[0]
        penalties = np.maximum(np.maximum(
            self.ranges[:, 0] - constraints_values_np, constraints_values_np - self.ranges[:, 1]), 0)
        return np.sum(np.power(penalties, self.beta) * self.factors[None, :], axis=-1)
