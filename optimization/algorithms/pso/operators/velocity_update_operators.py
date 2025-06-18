"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np

from optimization.algorithms.pso.types import Swarm
from optimization.algorithms.core import OptimizationStep


class InertiaVelocityUpdater(OptimizationStep):
    def __init__(self, c1=2.05, c2=2.05, inertia_w=1.8, inertia_decay_rate=0.8, vmaxs=None) -> None:
        super().__init__(self._do)
        self.c1 = c1
        self.c2 = c2
        self.inertia_w = inertia_w
        self.inertia_decay_rate = 0.8
        if vmaxs is not None:
            self.vmins = -vmaxs
            self.vmaxs = vmaxs

    def _do(self, swarm, algorithm, *args, **kwargs):
        if not hasattr(self, "vmaxs"):
            vmaxs = algorithm.optimization_problem.maxs / 2
            vmins = -vmaxs
        else:
            vmaxs = self.vmaxs
            vmins = self.vmins
        current_inertia_w: float = self.inertia_w * \
            (1 - self.inertia_decay_rate) ** algorithm.fes

        swarm.particles_velocities = current_inertia_w * swarm.particles_velocities + np.random.rand(swarm.n_individuals, swarm.n_vars) * self.c1 * (
            swarm.personal_bests - swarm.individuals) + np.random.rand(swarm.n_individuals, swarm.n_vars) * self.c2 * (swarm.personal_bests[swarm.global_bests] - swarm.individuals)
        swarm.particles_velocities = np.clip(
            swarm.particles_velocities, vmins, vmaxs)
        swarm.individuals += swarm.particles_velocities
        swarm.individuals = np.clip(
            swarm.individuals, algorithm.optimization_problem.mins, algorithm.optimization_problem.maxs)


class ConstrictionVelocityUpdater(OptimizationStep):
    def __init__(self, phi_max=4.1) -> None:
        super().__init__(self._do)
        self.phi_max = phi_max
        self.constriction_factor = 2 / np.abs(2 - self.phi_max -
                                              np.sqrt(np.square(self.phi_max) - 4 * self.phi_max))

    def _do(self, swarm, algorithm, *args, **kwargs):
        if swarm.global_bests.ndim == 1:
            swarm.global_bests = np.expand_dims(swarm.global_bests, axis=-1)
        nbsize: int = swarm.global_bests.shape[1]
        phis: np.ndarray[float] = np.random.uniform(
            0, self.phi_max / nbsize, (swarm.n_individuals, nbsize, swarm.n_vars))

        pm: np.ndarray[float] = np.sum(
            phis * swarm.personal_bests[swarm.global_bests], axis=1) / np.sum(phis, axis=1)

        swarm.particles_velocities = self.constriction_factor * (
            swarm.particles_velocities + np.sum(phis, axis=1) * (pm - swarm.individuals))

        swarm.individuals += swarm.particles_velocities
        swarm.individuals = np.clip(
            swarm.individuals, algorithm.optimization_problem.mins, algorithm.optimization_problem.maxs)


class EFVelocityUpdater(OptimizationStep):
    def __init__(self, phi_max=4.1) -> None:
        super().__init__(self._do)
        self.phi_max = phi_max
        self.constriction_factor = 2 / np.abs(2 - self.phi_max -
                                              np.sqrt(np.square(self.phi_max) - 4 * self.phi_max))

    def _do(self, swarm, algorithm, *args, **kwargs):

        # this shouldn't happen in ESPSO
        if swarm.global_bests.ndim == 1:
            swarm.global_bests = np.expand_dims(swarm.global_bests, axis=-1)

        # identify the largest and smallest species
        seeds, seeds_freq = np.unique(
            swarm.global_bests[:, 0], return_counts=True)
        seed_s = np.argmin(seeds_freq)
        size_s = seeds_freq[seed_s]
        seed_l = np.argmax(seeds_freq)
        size_l = seeds_freq[seed_l]

        dv = swarm.individuals[seed_s] - swarm.individuals[seed_l]
        ds = int(0.5 * (size_l - size_s))

        nbsize: int = swarm.global_bests.shape[1]
        phis: np.ndarray[float] = np.random.uniform(
            0, self.phi_max / nbsize, (swarm.n_individuals, nbsize, swarm.n_vars))

        pm: np.ndarray[float] = np.sum(
            phis * swarm.personal_bests[swarm.global_bests], axis=1) / np.sum(phis, axis=1)

        swarm.particles_velocities = self.constriction_factor * (
            swarm.particles_velocities + np.sum(phis, axis=1) * (pm - swarm.individuals))

       # identify particles which will be updated using the Equilibrium factor equation
        large_species_mask = swarm.global_bests[:, 0] == seed_l
        indices = np.arange(swarm.n_individuals)[large_species_mask][np.argsort(
            swarm.objective_values[large_species_mask][:ds])]
        # add the dv to particles that will be influenced by the EF
        swarm.particles_velocities[indices] += dv

        swarm.individuals += swarm.particles_velocities
        swarm.individuals = np.clip(
            swarm.individuals, algorithm.optimization_problem.mins, algorithm.optimization_problem.maxs)
