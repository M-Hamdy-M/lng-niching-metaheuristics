"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.spatial.distance import cdist
from optimization.algorithms.core import OptimizationStep


class SelectionPerformer(OptimizationStep):
    def __init__(self) -> None:
        super().__init__(self._do)

    def _do(self, population, algorithm, *args, **kwargs):
        distances = cdist(population.trial_vectors, population.individuals)
        nearest_ind = np.argmin(distances, axis=-1)

        for i in range(population.trial_vectors.shape[0]):
            if population.trial_objective_values[i] > population.objective_values[nearest_ind[i]]:
                population.individuals[nearest_ind[i]
                                       ] = population.trial_vectors[i].copy()
                population.objective_values[nearest_ind[i]
                                            ] = population.trial_objective_values[i].copy()
        population.trial_vectors = None
        population.trial_objective_values = None