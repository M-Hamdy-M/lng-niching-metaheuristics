"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

# this to prevent runtime error caused by enclosing class type hinting (OptimizationStep)
from __future__ import annotations
import typing


class OptimizationStep:
    """A step in an optimization algorithm pipeline.

    This class represents a step in an optimization algorithm pipeline. Each step encapsulates
    an operator that manipulates the population and algorithm state. Steps can be linked together
    to form a sequence of operations.
    """

    def __init__(self, operator) -> None:
        self.operator = operator
        self.next: typing.Optional[OptimizationStep] = None

    @property
    def next(self):
        """OptimizationStep: The next optimization step in the sequence."""
        return self.__next

    @next.setter
    def next(self, next_step) -> None:
        """Set the next optimization step in the sequence.

        Args:
            next_step (OptimizationStep): The next optimization step to link.

        Raises:
            Exception: If next_step is not an instance of OptimizationStep.

        """
        if next_step is None or isinstance(next_step, OptimizationStep):
            self.__next = next_step
        else:
            raise Exception(
                "Invalid next_step: next_step must be an instance of OptimizationStep")

    def has_next(self):
        """Check if the step has a subsequent step.

        Returns:
            bool: True if a subsequent step exists, False otherwise.

        """
        return bool(self._next)

    def run(self, population, algorithm, *args, **kwargs):
        """Execute the optimization step.

        Args:
            population: The population of individuals.
            algorithm: The optimization algorithm instance.
            *args: Additional positional arguments for the operator.
            **kwargs: Additional keyword arguments for the operator.
        """
        self.operator(population, algorithm, *args, **kwargs)

    def __call__(self, next_step: OptimizationStep):
        """Link the current step to the next step in the sequence.

        Args:
            next_step (OptimizationStep): The next optimization step to link.

        Returns:
            OptimizationStep: The current step with the next step linked.

        """
        next_step.next = self
        return self
