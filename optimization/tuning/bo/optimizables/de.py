"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
from optimization.tuning.bo.types.optimizable_algorithm import OptimizableAlgorithm
import ConfigSpace as CS
from optimization.algorithms.de.variants import get_clde, get_ncde
    
class OptimizableNCDE(OptimizableAlgorithm):
    def __init__(self, n_individuals, max_fes, name="NCDE", verbose=0, **kwargs):
        bp = {}
        super().__init__(algorithm_getter=get_ncde, base_params=bp, name=name, n_individuals=n_individuals, max_fes=max_fes, verbose=verbose)
        


    def _get_bounds(self):
        bounds = {
            "C": CS.UniformFloatHyperparameter(
                "C", lower=0, upper=1, default_value=0.1),
            "F": CS.UniformFloatHyperparameter(
                "F", lower=0.1, upper=1, default_value=0.9),
        }
        return bounds
    
class OptimizableCLDE(OptimizableAlgorithm):
    def __init__(self, n_individuals, max_fes, name="CLDE", verbose=0, bp={}, **kwargs):
        super().__init__(algorithm_getter=get_clde, base_params=bp, name=name, n_individuals=n_individuals, max_fes=max_fes, verbose=verbose)
        


    def _get_bounds(self):
        bounds = {
            "pc": CS.UniformFloatHyperparameter(
                "pc", lower=0, upper=1, default_value=0.1),
            "F": CS.UniformFloatHyperparameter(
                "F", lower=0.1, upper=1, default_value=0.9),
            "clearing_capacity": CS.UniformIntegerHyperparameter(
                "clearing_capacity", lower=1, upper=10, default_value=5),
        }
        return bounds
    