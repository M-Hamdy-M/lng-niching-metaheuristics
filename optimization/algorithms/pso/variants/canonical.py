"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

from optimization.algorithms.pso.operators.component_evaluators import GbestEvaluator, LbestEvaluator
from optimization.algorithms.pso.operators.velocity_update_operators import ConstrictionVelocityUpdater
from optimization.algorithms.core import OptimizationAlgorithm
from optimization.algorithms.pso.operators.initializers import SwarmRandomInitializer


# Implementation is based on the details in: Niching Without Niching Parameters: Particle Swarm Optimization Using a Ring Topology
def get_gbest_pso(
        name: str = "GBEST PSO",
        phi_max: float = 4.1,) -> OptimizationAlgorithm:
    """Create an optimization algorithm instance for Global Best Particle Swarm Optimization (PSO).

    Args:
        name (str, optional): The name of the optimization algorithm. Defaults to "gbest PSO".
        phi_max (float, optional): The maximum value of the constriction factor for velocity update. Defaults to 4.1.

    Returns:
        OptimizationAlgorithm: An instance of the optimization algorithm for GBEST PSO.

    """
    first_step = GbestEvaluator()
    ConstrictionVelocityUpdater(
        phi_max=phi_max)(first_step)
    return OptimizationAlgorithm(name=name, first_step=first_step, initializer=SwarmRandomInitializer())


# Implementation is based on the details in: Niching Without Niching Parameters: Particle Swarm Optimization Using a Ring Topology
def get_lbest_pso(
        name: str = None,
        phi_max: float = 4.1,
        neighborhood_size: int = 3) -> OptimizationAlgorithm:
    """Create an optimization algorithm instance for Local Best Particle Swarm Optimization (PSO).

    Args:
        name (str, optional): A name for the PSO algorithm. If not provided, it will be generated based on neighborhood size.
        phi_max (float, optional): The maximum value of the constriction factor for velocity update. Defaults to 4.1.
        neighborhood_size (int, optional): The size of the neighborhood for local best evaluation. Defaults to 3.

    Returns:
        OptimizationAlgorithm: An instance of the optimization algorithm for LBEST PSO.

    """
    if name is None:
        name = f"r{neighborhood_size}pso"
    first_step = LbestEvaluator(neighborhood_size=neighborhood_size)
    ConstrictionVelocityUpdater(
        phi_max=phi_max)(first_step)
    return OptimizationAlgorithm(name=name, first_step=first_step, initializer=SwarmRandomInitializer())
