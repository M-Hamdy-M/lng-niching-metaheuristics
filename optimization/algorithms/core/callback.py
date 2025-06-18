"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

from .population import Population


class Callback:
    """
    Base class for defining callback functions that are called at different stages
    of an optimization loop.

    Callbacks are used to add custom behavior to the optimization process without
    modifying the core optimization algorithm. Each callback function defined here
    can be overridden in derived classes to implement specific behaviors during
    different phases of the optimization loop.
    """

    def on_optimization_begin(self, population: Population, algorithm, *args, **kwargs) -> None:
        """
        Called at the beginning of the optimization process.

        This function can be overridden to perform any necessary setup or
        initialization before the optimization loop starts.

        Args:
            population (Population): The initial population of solutions.
            algorithm (OptimizationAlgorithm): The optimization algorithm being used.
        """
        pass

    def on_optimizatoin_end(self, population: Population, algorithm, *args, **kwargs) -> None:
        """
        Called at the end of the optimization process.

        This function can be overridden to perform any cleanup or finalization
        tasks after the optimization loop completes.

        Args:
            population (Population): The final population of solutions.
            algorithm (OptimizationAlgorithm): The optimization algorithm being used.
        """
        pass

    def on_generation_begin(self, population: Population, algorithm, *args, **kwargs) -> None:
        """
        Called at the beginning of each generation in the optimization process.

        This function can be overridden to implement behaviors that should occur
        at the start of each generation.

        Args:
            population (Population): The current population of solutions.
            algorithm (OptimizationAlgorithm): The optimization algorithm being used.
        """
        pass

    def on_generation_end(self, population: Population, algorithm, *args, **kwargs) -> None:
        """
        Called at the end of each generation in the optimization process.

        This function can be overridden to implement behaviors that should occur
        at the end of each generation.

        Args:
            population (Population): The current population of solutions.
            algorithm (OptimizationAlgorithm): The optimization algorithm being used.
        """
        pass

    def on_generation_end_pre_evalaution(self, population: Population, algorithm, *args, **kwargs) -> None:
        """
        Called at the end of each generation before generation evaluation is done in the optimization process.

        This function can be overridden to implement behaviors that should occur
        at the end of each generation and before the objective evaluation is done.

        Args:
            population (Population): The current population of solutions.
            algorithm (OptimizationAlgorithm): The optimization algorithm being used.
        """
        pass

    def on_step_end(self, population: Population, algorithm, *args, **kwargs) -> None:
        """
        Called at the end of each optimization step.

        This function can be overridden to implement behaviors that should occur
        at the end of each optimization step.

        Args:
            population (Population): The current population of solutions.
            algorithm (OptimizationAlgorithm): The optimization algorithm being used.
        """
        pass
