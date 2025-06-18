"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import typing
import random
import numpy as np
from tqdm.notebook import tqdm

from optimization.problems.types import OptimizationProblem
from optimization.algorithms.core.optimization_step import OptimizationStep
from optimization.algorithms.core.initializers import Initializer
from optimization.algorithms.core.population import Population
from optimization.algorithms.core.callback import Callback
from optimization.algorithms.core.initializers import Initializer, RandomInitializer


class OptimizationAlgorithm:

    def __init__(self,
                 name: str,
                 first_step: OptimizationStep,
                 initializer: typing.Optional[Initializer] = None,
                 do_generation_evaluation: bool = True,) -> None:
        """
        Initialize an OptimizationAlgorithm.

        Args:
            name (str): The name of the optimization algorithm.
            first_step (OptimizationStep): The first step of the optimization algorithm.
            initializer (Initializer, optional): The initializer for the population (default is RandomInitializer).
            do_generation_evaluation (bool): Whether to perform generation-wise evaluation.
        """
        self.name: str = name
        self.first_step: OptimizationStep = first_step

        if initializer is None:
            initializer = RandomInitializer()
        self.initializer: Initializer = initializer

        self.do_generation_evaluation: bool = do_generation_evaluation

    def compile(self,
                optimization_problem: OptimizationProblem,
                population: typing.Optional[Population] = None,
                random_state: int = None,
                initializer_kwargs={}, pipe=None):
        """
        Compile the optimization algorithm.

        Args:
            optimization_problem (OptimizationProblem): The optimization problem to be solved.
            population (Population, optional): The initial population (if not passed a population will be initialized using the initializer).
            random_state (int): The random seed for reproducibility (default is 42).
            initializer_kwargs (dict): Additional keyword arguments for the initializer.
        """
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(int(random_state))

        self.optimization_problem: OptimizationProblem = optimization_problem
        if population is not None:  # if population is passed set it
            self._population = population
        else:   # otherwise generate one using the passed initializer
            self._population = self.initializer.initialize(
                self.optimization_problem, **initializer_kwargs)

        self._fes = 0
        self.max_fes = 0
        self.curr_gen = 0
        self.converged = False
        self.pipe = pipe

    def evaluate(self, individuals=None):
        """
        Evaluate the fitness of individuals in the population.

        Args:
            individuals (np.ndarray, optional): The individuals to be evaluated (if not passed current population individuals will be evaluated).

        Returns:
            np.ndarray: The objective values for the individuals.
        """
        if individuals is None:
            individuals = self._population.individuals

        n_individuals = 1 if np.asarray(individuals).ndim == 1 else np.asarray(individuals).shape[0]

        if self._fes >= self.max_fes:
            raise TerminateOptimization(
                "Terminating Optimization Loop: max_fes is reached")
        self._fes += n_individuals
        if hasattr(self, "pipe") and self.pipe is not None:
            self.pipe.send({"type": "progress", "value": self._fes})
        return self.optimization_problem(
            individuals, curr_step=self.fes, max_step=self.max_fes)

    def check_convergence(self):
        """
        Check if the optimization has converged.

        Returns:
            bool: True if the optimization has converged, False otherwise.
        """
        # TODO: should check if the fitnesses or positions of the individuals are not changing by much
        self.converged = False
        return self.converged

    def optimize(self,
                 fes: int,
                 callbacks: list[Callback] = [], verbose=0):
        """
        Optimize the population for a specified number of function evaluations.

        Args:
            fes (int): The number of allowed function evaluations to perform.
            callbacks (list[Callback], optional): List of optimization callbacks (default is an empty list).
            verbose (int): Verbosity level (default is 0).
        """
        # if population is not initialized initialize it
        self.max_fes += fes

        if not hasattr(self._population, "objective_values"):
            self._population.objective_values = self.evaluate(
                self._population.individuals)

        # call preoptimization callbacks
        for callback in callbacks:
            callback.on_optimization_begin(self._population, self)

        pbar = tqdm(total=self.max_fes, leave=False,
                    desc=f"{self.name} optimizing {self.optimization_problem} Max FES: {self.max_fes}", disable=(verbose == 0), position=1)
        try:
            # start optimizing
            while self.fes < self.max_fes:
                pbar.update(self.fes - pbar.n)
                self.curr_gen += 1
                # call pregeneration callbacks
                for callback in callbacks:
                    callback.on_generation_begin(self._population, self)

                curr_step = self.first_step
                while curr_step is not None and not self.converged:
                    curr_step.run(population=self._population, algorithm=self)
                    curr_step = curr_step.next
                    # call poststep callbacks
                    for callback in callbacks:
                        callback.on_step_end(self._population, self)
                if self.do_generation_evaluation:
                    for callback in callbacks:
                        callback.on_generation_end_pre_evalaution(
                            self._population, self)
                    # After each generation evaluate individuals
                    self._population.objective_values = self.evaluate(
                        self._population.individuals)
                # call postgeneration callbacks
                for callback in callbacks:
                    callback.on_generation_end(self._population, self)
        # if any operator raised a TerminationOptimization exception the optimization loop will be terminated
        except TerminateOptimization as e:
            pass
        # call postoptimization callbacks
        for callback in callbacks:
            callback.on_optimizatoin_end(self._population, self)
        pbar.close()

    @property
    def population(self) -> Population | None:
        """
        Get the current population of the optimization algorithm.

        Returns:
            Population: The current population.
        """
        return self._population

    @property
    def fes(self) -> int:
        """
        Get the number of function evaluations (FES) performed by the optimization algorithm.

        Returns:
        int: The number of FES.
        """
        return self._fes


class TerminateOptimization(Exception):
    def __init__(self, message):
        """
        Initialize a TerminateOptimization exception.

        Args:
        message (str): The message indicating the reason for termination.
        """
        super().__init__(message)