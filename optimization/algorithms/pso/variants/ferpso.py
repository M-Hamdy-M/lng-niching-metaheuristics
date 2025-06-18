"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

from optimization.algorithms.pso.operators.component_evaluators import NbestFEREvaluator
from optimization.algorithms.pso.operators.velocity_update_operators import ConstrictionVelocityUpdater, EFVelocityUpdater
from optimization.algorithms.pso.operators.initializers import SwarmRandomInitializer
from optimization.algorithms.pso.operators.local_search_operators import PbestMutationLocalSearch
from optimization.algorithms.core import OptimizationAlgorithm


def get_ferpso(
        name: str = "FERPSO",
        phi_max: float = 4.1,) -> OptimizationAlgorithm:
    """Create an optimization algorithm instance for Fitness Euclidean distance Ratio Particle Swarm Optimization (FERPSO).

    Args:
        name (str, optional): The name of the optimization algorithm. Defaults to "FERPSO".
        phi_max (float, optional): The maximum value of the constriction factor for velocity update. Defaults to 4.1.

    Returns:
        OptimizationAlgorithm: An instance of the optimization algorithm for FERPSO.

    """
    first_step = NbestFEREvaluator()
    s = ConstrictionVelocityUpdater(
        phi_max=phi_max)(first_step)
    return OptimizationAlgorithm(name=name, first_step=first_step, initializer=SwarmRandomInitializer())

# Proposed in: Niching particle swarm optimization with local search for multi-modal optimization [https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Niching+particle+swarm+optimization+with+local+search+for+multi-modal+optimization&btnG=]


def get_ferpso_wls(
        name: str = "FERPSO LS",
        phi_max: float = 4.1,
        local_search_c: float = 2.05) -> OptimizationAlgorithm:
    """Create an optimization algorithm instance for Fitness Euclidean distance Ratio Particle Swarm Optimization with Local Search (FERPSO LS).

    Args:
        name (str, optional): The name of the optimization algorithm. Defaults to "FERPSO LS".
        phi_max (float, optional): The maximum value of the constriction factor for velocity update. Defaults to 4.1.
        local_search_c (float, optional): The local search constant for the Pbest Mutation step. Defaults to 2.05.

    Returns:
        OptimizationAlgorithm: An instance of the optimization algorithm for FERPSO LS.

    """
    first_step = NbestFEREvaluator()
    s = ConstrictionVelocityUpdater(
        phi_max=phi_max)(first_step)
    s = PbestMutationLocalSearch(c=local_search_c)(s)
    return OptimizationAlgorithm(name=name, first_step=first_step, initializer=SwarmRandomInitializer())


def get_eferpso(
        name: str = "E-FERPSO",
        phi_max: float = 4.1,
        local_search_c=2.05) -> OptimizationAlgorithm:
    """Create an optimization algorithm instance for Fitness Euclidean distance Ratio Particle Swarm Optimization (E-FERPSO).

    Args:
        name (str, optional): The name of the optimization algorithm. Defaults to "E-FERPSO".
        phi_max (float, optional): The maximum value of the constriction factor for velocity update. Defaults to 4.1.
        local_search_c (float, optional): The local search constant for the Pbest Mutation step. Defaults to 2.05.

    Returns:
        OptimizationAlgorithm: An instance of the optimization algorithm for SPSO.

    """
    first_step = NbestFEREvaluator()
    s = EFVelocityUpdater(
        phi_max=phi_max)(first_step)
    s = PbestMutationLocalSearch(c=local_search_c)(s)
    return OptimizationAlgorithm(name=name, first_step=first_step, initializer=SwarmRandomInitializer())
