"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from optimization.algorithms.pso.types import Swarm
from optimization.algorithms.core.initializers import Initializer
from optimization.problems.types import OptimizationProblem


class SwarmRandomInitializer(Initializer):
    def __init__(self) -> None:
        """
        Initialize a SwarmRandomInitializer object.

        This initializer is used to create a population for swarm optimization algorithms.
        """
        super().__init__()

    def initialize(self, optimization_problem: OptimizationProblem, n_individuals: int, *args, **kwargs) -> Swarm:
        """
        Initialize a swarm population for swarm-based optimization.

        Args:
            optimization_problem (OptimizationProblem): The optimization problem for which to initialize the swarm.
            n_individuals (int): The number of individuals (i.e., particles) in the swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Swarm: The initialized swarm population.
        """
        vmaxs = optimization_problem.maxs / 2
        vmins = -vmaxs
        vmaxs = vmaxs
        swarm = Swarm(n_individuals=n_individuals,
                      n_vars=optimization_problem.input_dimension)
        super().initialize(optimization_problem=optimization_problem,
                           population=swarm)
        swarm.individuals = np.random.uniform(
            optimization_problem.mins, optimization_problem.maxs, (n_individuals, optimization_problem.input_dimension))
        swarm.particles_velocities = np.random.uniform(
            vmins, vmaxs, (n_individuals, optimization_problem.input_dimension))
        return swarm
