"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
from optimization.tuning.bo.types.optimizable_algorithm import OptimizableAlgorithm
import ConfigSpace as CS
from optimization.algorithms.pso.variants import get_ferpso

class OptimizableFERPSO(OptimizableAlgorithm):
    def __init__(self, n_individuals, max_fes, name="FERPSO", verbose=0, **kwargs):
        bp = {}
        super().__init__(algorithm_getter=get_ferpso, base_params=bp, name=name, n_individuals=n_individuals, max_fes=max_fes, verbose=verbose)
        


    def _get_bounds(self):
        bounds = {
            "phi_max": CS.UniformFloatHyperparameter(
                "phi_max", lower=4, upper=6, default_value=4.1),
        }
        return bounds
    