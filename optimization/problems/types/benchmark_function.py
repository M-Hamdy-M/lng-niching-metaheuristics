"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy.spatial.distance import cdist

import typing
from .optimization_problem import OptimizationProblem


class BenchmarkFunction(OptimizationProblem):
    def __init__(self,
                 name: str,
                 bounds: np.ndarray[float],
                 is_maximization: bool,
                 global_optima: np.ndarray[float],
                 tolerance: float = None,
                 niche_radius: float = None,
                 evaluator: typing.Callable[[np.ndarray[float]], float] = None,
                 local_optima: np.ndarray[float] = None,
                 max_fitness_evaluations: typing.Optional[int] = None,) -> None:
        super().__init__(name=name,
                         evaluator=evaluator,
                         bounds=bounds,
                         is_maximization=is_maximization)
        self.global_optima: np.ndarray[float] = global_optima
        self.tolerance: float = tolerance
        self.local_optima: np.ndarray[float] = local_optima
        self.max_fitness_evaluations = max_fitness_evaluations
        self.niche_radius = niche_radius

    @property
    def global_optima(self) -> np.ndarray[float]:
        return self.__global_optima

    @global_optima.setter
    def global_optima(self, global_optima: np.ndarray[float]) -> None:
        tmp = np.asarray(global_optima)
        if tmp.ndim != 2 or tmp.shape[1] != self.input_dimension:
            raise ValueError(
                f"Invalid shape for global_optima expected shape (Any, {self.input_dimension}) but got {tmp.shape}")
        self.__global_optima = np.asarray(global_optima)
        self.__global_optima_fit: float = np.max(
            self(global_optima)) if self.is_maximization else np.min(self(global_optima))

    @property
    def global_optima_fit(self) -> float:
        return self.__global_optima_fit

    @property
    def local_optima(self) -> np.ndarray[float]:
        return self.__local_optima

    @local_optima.setter
    def local_optima(self, local_optima: np.ndarray[float]) -> None:
        if local_optima is None:
            self.__local_optima = None
            self.__local_optima_fits = None
            return

        tmp = np.asarray(local_optima)
        if tmp.ndim != 2 or tmp.shape[1] != self.input_dimension:
            raise ValueError(
                f"Invalid shape for local_optima expected shape (Any, {self.input_dimension}) but got {tmp.shape}")
        self.__local_optima: np.ndarray[float] = np.asarray(local_optima)
        self.__local_optima_fits: np.ndarray[float] = self(local_optima)

    @property
    def local_optima_fit(self) -> np.ndarray[float]:
        return self.__local_optima_fits

    @property
    def tolerance(self) -> float:
        return self.__tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        # if not isinstance(tolerance, (int, float)):
        #     raise TypeError(
        #         f"Tolerance must be a float or int, got {type(tolerance)}")
        self.__tolerance = tolerance


    def update_to_scaled(self):
        self.do_scale = True
        self.original_bounds = self.bounds
        self.bounds = np.full((self.input_dimension, 2), [0, 1])
        self.global_optima = self.scale(self.global_optima)

    def get_unique_right_solutions(self,
                                   solutions,
                                   unique_tolerance,
                                   right_tolerance,
                                   sorted_indices=None,
                                   distance_weights=None):
        """
        Find and return unique right solutions from a list of solutions.

        Args:
            solutions (array-like):
                List of solutions to be filtered.
            unique_tolerance (float):
                The tolerance level to consider two solutions as unique. Solutions with
                distances greater than this tolerance are considered unique.
            right_tolerance (float):
                The tolerance level to consider a right solution as found. If the distance
                between a right solution and the nearest unique solution is less than this
                tolerance, the right solution is considered found.
            sorted_indices (array-like, optional):
                Precomputed indices of sorted solutions. If not provided, solutions will
                be sorted in descending order of fitness values.
                Default is None.
            distance_weights (array-like, optional):
                Custom weights to adjust the importance of each dimension when calculating distances.
                If not provided, default weights are calculated based on component ranges.
                Default is None.

        Returns:
            numpy.ndarray:
                An array containing unique right solutions based on the given tolerances.
        """
        if distance_weights is None:
            distance_weights = 1 / (self.maxs - self.mins)
        unique_right = []
        unique_solutions = self.get_unique_solutions(
            solutions, unique_tolerance, sorted_indices=sorted_indices, distance_weights=distance_weights)
        for right_solution in self.global_optima:  # for each right solution ...
            # ... calculate distance between current right solution and all unique solutions
            distances = cdist(
                unique_solutions, [right_solution], w=distance_weights).flatten()
            # get the nearest unique solution to the current right solution
            min_idx = np.argmin(distances)
            # if distance is less than sol_tolerance then this right solution considered found
            if distances[min_idx] < right_tolerance:
                unique_right.append(unique_solutions[min_idx])
        return np.array(unique_right)
