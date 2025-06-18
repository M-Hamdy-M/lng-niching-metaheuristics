"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Shahd Gaben & Mohamed Hamdy                    ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.spatial.distance import euclidean
import random as rand

from optimization.algorithms.core import OptimizationStep, OptimizationAlgorithm



class CLDEStep(OptimizationStep):
    def __init__(self, F=0.9, pc=0.1, strategy='DE/rand/1/bin', selection='select_best', tournament_size=3,
                 clearing_radius=0.25, clearing_capacity=3) -> None:
        super().__init__(self._do)
        self.F = F
        self.pc = pc
        self.variation = strategy
        self.selection = selection
        self.tournament_size = tournament_size
        self.clearing_radius = clearing_radius
        self.clearing_capacity = int(clearing_capacity)

    def _mutation_func(self, pop, i, best_sol, a, b, c, d, e):
        if self.variation == 'DE/rand/1/bin':
            return pop[a] + self.F * (pop[b] - pop[c])
        elif self.variation == 'DE/best/1/bin':
            return best_sol + self.F * (pop[b] - pop[c])
        elif self.variation == 'DE/rand-to-best/1/bin':
            return pop[i] + self.F * (best_sol - pop[i]) + self.F * (pop[b] - pop[c])
        elif self.variation == 'DE/rand/2/bin':
            return pop[a] + self.F * (pop[b] - pop[c]) + self.F * (pop[d] - pop[e])
        elif self.variation == 'DE/best/2/bin':
            return best_sol + self.F * (pop[b] - pop[c]) + self.F * (pop[d] - pop[e])

    def _do(self, population, algorithm: OptimizationAlgorithm, *args, **kwargs):

        # Determine the mutation function based on the chosen strategy

        # recombination loop
        trial_pop = np.copy(population.individuals)
        best_sol = population.individuals[np.argmax(
            population.objective_values)]  # maximization specific

        # Mutation loop
        for i in range(population.n_individuals):
            pop_indx = list(range(population.n_individuals))
            pop_indx.remove(i)

            i1, i2, i3, i4, i5 = rand.sample(pop_indx, 5)

            mutant = self._mutation_func(
                            population.individuals, i, best_sol, i1, i2, i3, i4, i5)
            for j in range(algorithm.optimization_problem.input_dimension):
                if rand.random() < self.pc:
                    trial_pop[i][j] = np.clip(
                        mutant[j],
                        algorithm.optimization_problem.mins[j],
                        algorithm.optimization_problem.maxs[j]
                    )

            # If trial vector is unchanged, perform a random mutation
            if np.all(trial_pop[i] == population.individuals[i]):
                n = rand.randint(
                    0, algorithm.optimization_problem.input_dimension - 1)
                trial_pop[i][n] = np.clip(
                    mutant[n],
                    algorithm.optimization_problem.mins[n],
                    algorithm.optimization_problem.maxs[n]
                )

        trial_fit = algorithm.evaluate(trial_pop)

        # fitness sharing
        mu_plus_lambda_pop = np.concatenate(
            (population.individuals, trial_pop))
        mu_plus_lambda_fit = np.concatenate(
            (population.objective_values, trial_fit))

        sorted_mu_lambda_pop = mu_plus_lambda_pop[np.argsort(
            mu_plus_lambda_fit)][::-1]  # maximization specific
        sorted_mu_lambda_fit = np.sort(mu_plus_lambda_fit)[
            ::-1]  # maximization specific

        sharing_pop = []
        cleared_fit = []
        original_fit = []

        while sorted_mu_lambda_pop.shape[0] > 0:
            distances = np.array([euclidean(sorted_mu_lambda_pop[0], sorted_mu_lambda_pop[j])
                                 for j in range(sorted_mu_lambda_pop.shape[0])])
            in_niche = distances < self.clearing_radius
            sharing_pop.extend(sorted_mu_lambda_pop[in_niche])
            original_fit.extend(sorted_mu_lambda_fit[in_niche])

            if np.sum(in_niche) > self.clearing_capacity:
                # maximization specific
                sorted_mu_lambda_fit[np.where(
                    in_niche)[0][self.clearing_capacity:]] = -np.inf

            cleared_fit.extend(sorted_mu_lambda_fit[in_niche])

            sorted_mu_lambda_pop = sorted_mu_lambda_pop[np.logical_not(
                in_niche)]
            sorted_mu_lambda_fit = sorted_mu_lambda_fit[np.logical_not(
                in_niche)]

        sharing_pop = np.array(sharing_pop)
        original_fit = np.array(original_fit)
        cleared_fit = np.array(cleared_fit)

        # Selection
        if self.selection == 'tournament':
            selected_indices = []
            for _ in range(population.n_individuals):
                indices = list(range(sharing_pop.shape[0]))
                tournament_indices = rand.sample(indices, self.tournament_size)
                tournament_fit = cleared_fit[tournament_indices]
                winner_index = tournament_indices[np.argmax(
                    tournament_fit)]  # Maximization specific
                selected_indices.append(winner_index)
            selected_pop, selected_fit = sharing_pop[selected_indices], original_fit[selected_indices]

        elif self.selection == 'select_best':
            selected_pop = sharing_pop[np.argsort(
                cleared_fit)][::-1][:population.n_individuals]
            selected_fit = original_fit[np.argsort(
                cleared_fit)][::-1][:population.n_individuals]

        population.individuals[:] = selected_pop.copy()
        population.objective_values[:] = selected_fit.copy()


def get_clde(name="CLDE", F=0.9, pc=0.1, strategy='DE/best/1/bin', selection='select_best', tournament_size=3,
             clearing_radius=0.25, clearing_capacity=3):
    first_step = CLDEStep(F=F, pc=pc, strategy=strategy, selection=selection, tournament_size=tournament_size,
                          clearing_radius=clearing_radius, clearing_capacity=clearing_capacity)

    return OptimizationAlgorithm(
        name=name,
        first_step=first_step,
        do_generation_evaluation=False
    )