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


class GbestEvaluator(OptimizationStep):
    def __init__(self) -> None:
        super().__init__(self._do)

    def _do(self, swarm: Swarm, algorithm: OptimizationAlgorithm, *args, **kwargs):
        swarm.global_bests = np.stack([np.full((swarm.n_individuals, ), np.argmax(
            swarm.personal_bests_fitnesses), np.int32), np.arange(swarm.n_individuals)], axis=-1)


class LbestEvaluator(OptimizationStep):
    def __init__(self, neighborhood_size=3) -> None:
        super().__init__(self._do)
        if neighborhood_size <= 1:
            raise ValueError(
                "Invalid value for neighborhood_size, must be more than 1")
        self.neighborhood_size = neighborhood_size

    def _do(self, swarm: Swarm, algorithm: OptimizationAlgorithm, *args, **kwargs):
        swarm.global_bests = np.stack([
            (
                np.argmax(
                    np.lib.stride_tricks.sliding_window_view(
                        np.concatenate([
                            swarm.personal_bests_fitnesses[-int(
                                np.ceil((self.neighborhood_size - 1) / 2)):],
                            swarm.personal_bests_fitnesses,
                            swarm.personal_bests_fitnesses[:int(
                                np.floor((self.neighborhood_size - 1) / 2))]
                        ]),
                        self.neighborhood_size),
                    -1
                ) - np.floor((self.neighborhood_size - 1) / 2) + np.arange(len(swarm.personal_bests_fitnesses))
            ) % len(swarm.personal_bests_fitnesses),
            np.arange(swarm.n_individuals)], axis=-1)


class NbestSpatialDistanceEvaluator(OptimizationStep):
    def __init__(self, nbsize_range: list[int] = [2, 5], distance_metric: str = "euclidean") -> None:
        super().__init__(self._do)
        self.nbsize_range = nbsize_range
        self.distance_metric = distance_metric

    def _do(self, swarm: Swarm, algorithm: OptimizationAlgorithm, *args, **kwargs):
        current_nbsize = int(np.ceil(
            (algorithm.fes/algorithm.max_fes) * (self.nbsize_range[1] - 1)) + self.nbsize_range[0] - 1)

        swarm.global_bests = np.argsort(cdist(
            swarm.personal_bests, swarm.personal_bests, metric=self.distance_metric), axis=-1)[:, :current_nbsize]


class NbestFEREvaluator(OptimizationStep):
    def __init__(self) -> None:
        super().__init__(self._do)
        # params["include_pbest"] = True

    def get_fer_nbests(self, swarm, diagonal) -> np.ndarray[int]:
        range = np.ptp(swarm.personal_bests_fitnesses)
        alpha = diagonal/(range) if range != 0 else 0
        distances = cdist(swarm.personal_bests, swarm.personal_bests)
        fit_diff = swarm.personal_bests_fitnesses - \
            np.tile(np.array([swarm.personal_bests_fitnesses]
                             ).T, (1, swarm.n_individuals))
        mask = (distances == 0)
        fer = np.full((swarm.n_individuals, swarm.n_individuals), -np.inf)
        fer[np.logical_not(mask)] = fit_diff[np.logical_not(
            mask)] / distances[np.logical_not(mask)]
        fer *= alpha
        nbests = np.argmax(fer, axis=-1)
        return nbests

    def _do(self, swarm: Swarm, algorithm: OptimizationAlgorithm, *args, **kwargs):
        if not hasattr(self, "diagonal_length"):
            self.diagonal_length = np.linalg.norm(
                algorithm.optimization_problem.mins - algorithm.optimization_problem.maxs)
        # swarm.global_bests = self.get_fer_nbests(swarm, self.diagonal_length)
        swarm.global_bests = np.stack([self.get_fer_nbests(
            swarm, self.diagonal_length), np.arange(swarm.n_individuals)], axis=-1)


class LbestSPSOEvaluator(OptimizationStep):
    def __init__(self, species_radius=None) -> None:
        super().__init__(self._do)
        self.species_radius = species_radius

    def get_spso_lbests(self, particles, fitnesses, species_radius) -> np.ndarray[int]:
        sorted_indices = np.argsort(fitnesses)
        seeds = set()
        lbests = np.empty((particles.shape[0],), dtype=np.uint16)
        for i in range(particles.shape[0]):
            current_particle = particles[sorted_indices[i]]
            found = False
            for seed in seeds:
                distance = np.linalg.norm(current_particle - particles[seed])
                if distance <= species_radius:  # if the particle is within any species seed, ...
                    found = True    # ... found flag is set to true to indicate the particle belong to an existing species
                    # lbest for this particle is set to current species seed
                    lbests[sorted_indices[i]] = seed
                    break   # we break from the loop to avoid unneccerary iterations

            # if current particles is not in the readius of any species seed ...
            if (not found):
                seeds.add(i)  # ... add it as a new species seed
                # the local best for the species seed is itself
                lbests[sorted_indices[i]] = sorted_indices[i]
        return lbests

    def _do(self, swarm: Swarm, algorithm: OptimizationAlgorithm, *args, **kwargs):

        if not hasattr(self, "species_radius") or self.species_radius is None:
            self.species_radius = np.linalg.norm(
                algorithm.optimization_problem.maxs - algorithm.optimization_problem.mins) / 10

        swarm.global_bests = np.stack([self.get_spso_lbests(
            swarm.individuals, swarm.objective_values, self.species_radius), np.arange(swarm.n_individuals)], axis=-1)
