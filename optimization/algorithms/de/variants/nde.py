"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy & Shahd Gaben                    ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.spatial.distance import cdist
from optimization.algorithms.core import OptimizationAlgorithm, OptimizationStep
from optimization.algorithms.core import OptimizationAlgorithm, OptimizationStep
from optimization.algorithms.common.selection import SelectionPerformer


class DENeigbourhoodRecombination(OptimizationStep):
    def __init__(self, F: float, C: float, m=None) -> None:
        super().__init__(self._do)
        self.F = F
        self.C = C
        self.m = m

    def _do(self, population, algorithm: OptimizationAlgorithm, *args, **kwargs):
        if not hasattr(population, "trial_vectors") or population.trial_vectors is None:
            population.trial_vectors = population.individuals.copy()
        if self.m is not None:
            curr_m = self.m
        if population.n_individuals <= 200:
            curr_m = int(
                5+5*((algorithm.max_fes-algorithm.fes)/algorithm.max_fes))
        else:
            curr_m = int(
                20+30*((algorithm.max_fes-algorithm.fes)/algorithm.max_fes))
        neighbours = np.argsort(
            cdist(population.trial_vectors, population.trial_vectors), axis=-1)[:, :curr_m]

        # Perform Mutation and crossover
        rs = np.array([neighbours[i][[r if r != i else neighbours.shape[1] -
                                      1 for r in np.random.choice(neighbours.shape[1] - 1, 3, replace=False)]] for i in range(population.trial_vectors.shape[0])])
        # perform mutation
        mutants = population.trial_vectors[rs[:, 0]] + self.F * (
            population.trial_vectors[rs[:, 1]] - population.trial_vectors[rs[:, 2]])
        # perform crossover
        rcs = np.random.rand(
            population.trial_vectors.shape[0], algorithm.optimization_problem.input_dimension)
        jrs = np.random.choice(
            algorithm.optimization_problem.input_dimension, population.trial_vectors.shape[0])

        # create mutation mask
        mutation_masks = rcs < self.C
        mutation_masks[np.arange(
            population.trial_vectors.shape[0]), jrs] = True
        # produce the trial vector
        population.trial_vectors[mutation_masks] = mutants[mutation_masks]
        population.trial_vectors = np.clip(
            population.trial_vectors, algorithm.optimization_problem.mins, algorithm.optimization_problem.maxs)
        population.trial_objective_values = algorithm.evaluate(
            population.trial_vectors)

def get_ncde(
        F: float,
        C: float,
        m=None,
        name: str = "NCDE",
) -> OptimizationAlgorithm:
    """Create an optimization algorithm instance for Neighbourhood Crowding Differential Evolution (NCDE).
    Args:
        F (float): Scaling factor for differential mutation.
        C (float): Crossover rate.
        m (Optional[int]): Number of nearest neighbors to consider. If not passed will be dynamically set based on n_individuals.
        name (str, optional): Name of the algorithm. Default is "NCDE".

    Returns:
        OptimizationAlgorithm: An instance of the optimization algorithm for NCDE.

    """
    first_step = DENeigbourhoodRecombination(F=F, C=C, m=m)
    s = SelectionPerformer()(first_step)
    return OptimizationAlgorithm(
        name=name,
        first_step=first_step,
        do_generation_evaluation=False,
    )
