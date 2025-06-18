"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

from optimization.algorithms.pso.operators.component_evaluators import  NbestSpatialDistanceEvaluator
from optimization.algorithms.pso.operators.velocity_update_operators import  ConstrictionVelocityUpdater
from optimization.algorithms.pso.operators.initializers import SwarmRandomInitializer
from optimization.algorithms.core import OptimizationAlgorithm


def get_lipso(
        name: str = "LIPSO",
        phi_max: float = 4.1,
        nbsize_range: list[int] = [2, 5],
        distance_metric: str = "euclidean") -> OptimizationAlgorithm:
    """Create an optimization algorithm instance for Locally Informed Particle Swarm (LIPS).

    Args:
        name (str, optional): The name of the optimization algorithm. Defaults to "LIPS".
        phi_max (float, optional): The maximum value of the constriction factor for velocity update. Defaults to 4.1.
        nbsize_range (list[int], optional): Range of neighborhood sizes for local information evaluation. Defaults to [2, 5].
        distance_metric (str, optional): The distance metric used for spatial information computation. Defaults to "euclidean".

    Returns:
        OptimizationAlgorithm: An instance of the optimization algorithm for LIPS.

    """
    first_step = NbestSpatialDistanceEvaluator(
        nbsize_range=nbsize_range, distance_metric=distance_metric)
    s = ConstrictionVelocityUpdater(
        phi_max=phi_max)(first_step)
    return OptimizationAlgorithm(name=name, first_step=first_step, initializer=SwarmRandomInitializer())
