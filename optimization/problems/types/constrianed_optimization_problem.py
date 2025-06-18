"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import numpy as np
import typing
from optimization.problems.types.constraint_handlers import PenaltyConstraintHandler, DefaultPenaltyConstraintHandler
from optimization.problems.types.optimization_problem import OptimizationProblem


class ConstrainedOptimizationProblem(OptimizationProblem):
    def __init__(self,
                 name: str,
                 bounds: np.ndarray[float],
                 is_maximization: bool,
                 evaluator: typing.Callable,
                 do_scale: bool = False,
                 constraint_handler: PenaltyConstraintHandler = None, variable_names=None,constraint_names=None) -> None:
        super().__init__(name=name, bounds=bounds, is_maximization=is_maximization,
                         evaluator=evaluator, do_scale=do_scale, variable_names=variable_names)

        if constraint_handler is None:
            constraint_handler = DefaultPenaltyConstraintHandler()
        self.constraint_handler = constraint_handler
        self.constraint_names = constraint_names
        
        
    def evaluate(self, individuals: list[list[float]], **kwargs):
        individuals = self._validate_individuals(individuals=individuals)

        objs, constraints = self.evaluator(individuals)
        objs = (-1 if not self.is_maximization else 1) * objs
        return objs, constraints, self._evaluate_penalized_obj(objs, constraints, **kwargs)

    def __call__(self, individuals: list[list[float]], **kwargs) -> float:
        _, _, penalized_objective_values = self.evaluate(individuals)
        self.fes += len(individuals)
        return penalized_objective_values

    def _evaluate_penalized_obj(self, objective_values, constraints, **kwargs):
        penalties = self.constraint_handler(
            constraints_values=constraints, **kwargs)
        return objective_values - penalties
    
    @property
    def n_constraints(self):
        try:
            return len(self.constraint_handler.constraints)
        except Exception:
            return 0