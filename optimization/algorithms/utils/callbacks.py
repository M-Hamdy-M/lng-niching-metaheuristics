"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
from optimization.algorithms.core.callback import Callback
from optimization.algorithms.core import Population
import typing


class SolutionTracer(Callback):
    def __init__(self, population_callbacks: dict[str, typing.Callable]) -> None:
        super().__init__()
        self.history = {}
        self.callbacks = population_callbacks
        for callback_title in self.callbacks.keys():
                self.history[callback_title] = []
    
    def on_generation_end(self, population: Population, algorithm, *args, **kwargs) -> None:
        for callback_title, callback in self.callbacks.items():
            self.history[callback_title].append(callback(population))

    def on_optimizatoin_end(self, population: Population, algorithm, *args, **kwargs) -> None:
        algorithm.call_back_history = self.history