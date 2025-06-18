"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

from abc import ABC, abstractmethod
import numpy as np

from optimization.algorithms.core.population import Population
from optimization.problems.types import OptimizationProblem


class Initializer(ABC):

    def __init__(self) -> None:
        """
        Initialize an Initializer object.

        This is an abstract base class for initializing populations before optimization.
        """
        pass

    @abstractmethod
    def initialize(self, optimization_problem: OptimizationProblem, *args, **kwargs):
        """
        Initialize a population of individuals.

        Args:
            optimization_problem (OptimizationProblem): The optimization problem for which to initialize the population.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Population: The initialized population of individuals.
        """
        pass

class RandomInitializer(Initializer):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, optimization_problem: OptimizationProblem, n_individuals: int, *args, **kwargs) -> Population:
        """
        Initialize a population of individuals with random values.

        Args:
            optimization_problem (OptimizationProblem): The optimization problem for which to initialize the population.
            n_individuals (int): The number of individuals to initialize.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Population: The initialized population of individuals.
        """
        population = Population()
        super().initialize(optimization_problem=optimization_problem,
                           population=population)
        population.individuals = np.random.uniform(
            optimization_problem.mins, optimization_problem.maxs, (n_individuals, optimization_problem.input_dimension))
        return population

