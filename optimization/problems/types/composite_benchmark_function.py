"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import typing
from .benchmark_function import BenchmarkFunction

import numpy as np


class CompositeBenchmarkFunction(BenchmarkFunction):
    def __init__(self,
                 name: str,
                 bounds: np.ndarray[float, typing.Any],
                 is_maximization: bool,
                 global_optima: np.ndarray[float, typing.Any],
                 basic_functions: list[typing.Callable],
                 lambdas,
                 tolerance: float = None,
                 niche_radius: float = None,
                 sigmas=None,
                 ms=None,
                 biases=None,
                 f_bias=0,
                 C=2000,
                 local_optima: np.ndarray[float, typing.Any] = None) -> None:
        self.bounds = bounds
        self.basic_functions = basic_functions
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.biases = biases
        self.f_bias = f_bias
        self.ms = ms
        self.C = C
        super().__init__(name=name,
                         evaluator=self._evaluate,
                         bounds=bounds,
                         is_maximization=is_maximization,
                         global_optima=global_optima,
                         tolerance=tolerance,
                         niche_radius=niche_radius,
                         local_optima=local_optima)

    @property
    def N(self) -> int:
        return self.__N

    @property
    def basic_functions(self):
        return self.__basic_functions

    @basic_functions.setter
    def basic_functions(self, basic_functions: list[typing.Callable]):
        for i, basic_function in enumerate(basic_functions):
            if not callable(basic_function):
                raise TypeError(
                    f"Expected a list of callables but got a non callable type at index {i}")
        self.__basic_functions = basic_functions
        self.__N: int = len(basic_functions)

    @property
    def biases(self):
        return self.__biases

    @biases.setter
    def biases(self, biases):
        if biases is None:
            if self.N is None:
                raise ValueError("Biases cannot be None!")
            else:
                self.__biases = np.zeros((self.N, ))
                return
        tmp = np.asarray(biases)
        if tmp.shape == (self.N, ):
            self.__biases = biases
        else:
            raise ValueError(
                f"Invalid shape for biases expected ({self.N}, ) but got {tmp.shape}")

    @property
    def ms(self):
        return self.__ms

    @ms.setter
    def ms(self, ms):
        if ms is None:
            if self.N is None or self.input_dimension is None:
                raise ValueError("M cannot be None!")
            else:
                self.__ms = np.tile(np.expand_dims(
                    np.eye(self.input_dimension), axis=0), (self.N, 1, 1))
                return
        tmp = np.asarray(ms)
        if tmp.shape == (self.N, self.input_dimension, self.input_dimension):
            self.__ms = ms
        else:
            raise ValueError(
                f"Invalid shape for M expected {(self.N, self.input_dimension, self.input_dimension)} but got {tmp.shape}")

    @property
    def sigmas(self):
        return self.__sigmas

    @sigmas.setter
    def sigmas(self, sigmas):
        if sigmas is None:
            if self.N is None:
                raise ValueError("Sigmas cannot be None!")
            else:
                self.__sigmas = np.ones(self.N)
                return
        tmp = np.asarray(sigmas)
        if tmp.shape == (self.N, ):
            self.__sigmas = sigmas
        else:
            raise ValueError(
                f"Invalid shape for sigmas expected {(self.N, )} but got {tmp.shape}")

    def _evaluate(self, X):
        ws = np.empty((self.N, X.shape[0]))
        fits = np.empty((self.N, X.shape[0]))

        y = np.full((self.input_dimension, ), 5)
        fmax = np.array([self.basic_functions[i](np.expand_dims(
            np.dot(y, self.ms[i]) / self.lambdas[i], 0)) for i in range(self.N)])

        for i in range(self.N):
            ws[i] = np.exp(-(np.sum(np.square(X - self.global_optima[i]), axis=-1)) /
                            (2 * self.input_dimension * np.square(self.sigmas[i])))
            fits[i] = self.basic_functions[i](
                np.divide(np.dot((X - self.global_optima[i]), self.ms[i]), self.lambdas[i]))

        maxs = np.max(ws, axis=0)
        for i in range(self.N):
            ws[i][ws[i] < maxs] *= (1 - np.power(maxs[ws[i] < maxs], 10))

        ws = ws / np.sum(ws, axis=0)
        fits = self.C * fits / fmax
        return np.sum(ws * (fits + np.expand_dims(self.biases, axis=-1)), axis=0) + self.f_bias
