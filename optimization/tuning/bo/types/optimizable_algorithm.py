"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
from abc import ABC, abstractmethod
import numpy as np
import random 
from optimization.problems.types.constrianed_optimization_problem import ConstrainedOptimizationProblem


class OptimizableAlgorithm(ABC):
    def __init__(self, 
            algorithm_getter, 
            base_params,
            name,
            n_individuals,
            max_fes=np.inf,
            verbose=0,
        ):
        self.algorithm_getter = algorithm_getter
        self.base_params = base_params
        self.name = name
        self.n_individuals = n_individuals
        self.max_fes = max_fes
        self.verbose = verbose
        

    @abstractmethod
    def _get_bounds(self):
        pass


    def _get_extra_params(self):
        return {}

    def get_bounds(self):
        return self._get_bounds()
        

    def compile(self, problem):
        self.problem = problem


    def gen_ran_arr(self, size, seed,  max=1000):
        random.seed(seed)
        arr = np.array([int(random.random() * max) for i in range(size)], dtype=int)
        return arr
    
    def evaluate(self, config, seed, n_times, return_algorithms=False, kwargs={}):

        if n_times == 1:
            seeds = [seed]
        else:
            seeds = self.gen_ran_arr(n_times, seed)
        params = self.base_params | self._get_extra_params() | dict(config)
        best_fitnesses = []
        algorithms = []
        for rep_idx in range(n_times):
            algorithm = self.algorithm_getter(**params)
            algorithm.compile(optimization_problem=self.problem,
                            initializer_kwargs={"n_individuals": self.n_individuals}, random_state=seeds[rep_idx])
            
            algorithm.optimize(fes=self.max_fes, verbose=self.verbose, **kwargs)

            if isinstance(self.problem, ConstrainedOptimizationProblem):
                _, _, objs = self.problem.evaluate(algorithm.population.solutions)
            else:
                objs = self.problem.evaluate(algorithm.population.solutions)
            best_fitnesses.append(np.max(objs))
            if return_algorithms:
                algorithms.append(algorithm)
        if return_algorithms:
            return best_fitnesses, algorithms
        return best_fitnesses
        
        
    def evaluate_cost(self, config, seed=42):
        if self.verbose > 1:
            print("Trying: ", config)
        # config_dict = dict(config)
        params = self.base_params | self._get_extra_params() | dict(config)
        algorithm = self.algorithm_getter(**params)

        algorithm.compile(optimization_problem=self.problem,
                          initializer_kwargs={"n_individuals": self.n_individuals}, random_state=seed)
        algorithm.optimize(fes=self.max_fes, callbacks=[], verbose=1)

        if isinstance(self.problem, ConstrainedOptimizationProblem):
            _, _, objs = self.problem.evaluate(algorithm.population.solutions)
        else:
            objs = self.problem.evaluate(algorithm.population.solutions)
        best_fitness = np.max(objs)
        cost = -best_fitness 
        if self.verbose > 1:
            print("Cost: ", cost)
        return cost
        