"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.spatial.distance import cdist

from optimization.algorithms.pso.types import Swarm
from optimization.algorithms.core import OptimizationStep, OptimizationAlgorithm

class PbestMutationLocalSearch(OptimizationStep):
    def __init__(self, c: float = 2.05) -> None:
        super().__init__(self._do)
        self.c = c

    def _do(self, swarm: Swarm, algorithm: OptimizationAlgorithm, *args, **kwargs):
        distances = cdist(swarm.personal_bests, swarm.personal_bests)
        nearest_pbests = distances.argsort(axis=-1)[:, 1]
        tmp = np.empty(swarm.personal_bests.shape)

        mask = swarm.personal_bests_fitnesses[nearest_pbests] >= swarm.personal_bests_fitnesses
        tmp[mask] = swarm.personal_bests[mask] + np.random.uniform(0, self.c, swarm.personal_bests[mask].shape) * (
            swarm.personal_bests[nearest_pbests][mask] - swarm.personal_bests[mask])
        tmp[np.logical_not(mask)] = swarm.personal_bests[np.logical_not(mask)] + np.random.uniform(0, self.c, swarm.personal_bests[np.logical_not(
            mask)].shape) * (swarm.personal_bests[np.logical_not(mask)] - swarm.personal_bests[nearest_pbests][np.logical_not(mask)])

        tmp = np.clip(
            tmp, algorithm.optimization_problem.mins, algorithm.optimization_problem.maxs)
        tmp_fits = algorithm.evaluate(tmp)

        mask = tmp_fits > swarm.personal_bests_fitnesses
        swarm.personal_bests[mask] = tmp[mask]
        swarm.personal_bests_fitnesses[mask] = tmp_fits[mask]