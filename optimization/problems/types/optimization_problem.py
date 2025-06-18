"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import numpy as np
import typing
from scipy.spatial.distance import cdist


class OptimizationProblem:
    def __init__(self,
                 name: str,
                 bounds: np.ndarray[float],
                 is_maximization: bool,
                 evaluator: typing.Callable = None,
                 do_scale: bool = False, 
                 variable_names=None,) -> None:
        self.name: str = name
        self.bounds: np.ndarary[float] = np.array(bounds)
        self.is_maximization: bool = is_maximization

        self.evaluator = evaluator
        self.do_scale = bool(do_scale)
        self.original_bounds = np.array(bounds)
        if do_scale:
            ## keep the original_bounds to be used for scaling and inverse scaling
            ## set the bounds to be between [0, 1]
            self.bounds = np.full((self.input_dimension, 2), [0, 1])
            
        self.fes = 0
        if variable_names is None:
            variable_names = [f"x_{i}" for i in range(self.input_dimension)]
        elif len(variable_names) != self.input_dimension:
            raise ValueError(f"Invalid length for variable_names. Expected {self.input_dimnesion} but got {len(variable_names)}")
        self.variable_names = variable_names

        
    def evaluate(self, individuals: list[list[float]], **kwargs):
        individuals = self._validate_individuals(individuals=individuals)
        objs = self.evaluator(individuals)
        self.fes += len(individuals)
        objs = (-1 if not self.is_maximization else 1) * objs
        return objs

    def __call__(self, individuals: list[list[float]], **kwargs) -> float:
        return self.evaluate(individuals)

    @property
    def bounds(self) -> np.ndarray[float]:
        return self.__bounds

    @bounds.setter
    def bounds(self, bounds) -> None:
        tmp = np.asarray(bounds)
        if tmp.ndim != 2 or tmp.shape[1] != 2:
            raise ValueError(
                f"Invalid shape expected shape (Any, 2) but got {tmp.shape}")
        self.__bounds = bounds
        self.__input_dimension: int = bounds.shape[0]

    @property
    def mins(self) -> np.ndarray[float]:
        return self.bounds[:, 0]

    @property
    def maxs(self) -> np.ndarray[float]:
        return self.bounds[:, 1]

    @property
    def input_dimension(self) -> int:
        return self.__input_dimension

    def scale(self, individuals):
        if not self.do_scale:
            print("[Warning]: Trying to scale while do_scale is False")
            return individuals
        return (individuals - self.original_bounds[:, 0]) / (self.original_bounds[:, 1] - self.original_bounds[:, 0])

    def inverse_scale(self, individuals):
        if not self.do_scale:
            print("[Warning]: Trying to inverse scale while do_scale is False")
            return individuals
        return individuals * (self.original_bounds[:, 1] - self.original_bounds[:, 0]) + self.original_bounds[:, 0]

    def _validate_individuals(self, individuals):
        if not ((np.asarray(individuals).ndim == 1 and np.asarray(individuals).shape[0] == self.input_dimension) or (np.asarray(individuals).ndim == 2 and np.asarray(individuals).shape[1] == self.input_dimension)):
            raise ValueError(
                f"Invalid Shape for Individuals: expected shape (Any, {self.input_dimension}) or ({self.input_dimension}, ), but got {np.asarray(individuals).shape}")
        individuals = np.asarray(individuals)
        if individuals.ndim == 1:
            individuals = np.array([individuals])
        if self.do_scale:
            return self.inverse_scale(individuals)
        return individuals

    def get_unique_solutions_indices(self, solutions, tolerance=1.0, sorted_indices=None, distance_weights=None):
        """
        Find unique solution indices based on a given tolerance, considering custom distance weights.

        Args:
            solutions (array-like):
                List of solutions to be evaluated.
            tolerance (float, optional):
                The tolerance level to consider two solutions as unique. Solutions with
                distances greater than this tolerance are considered unique.
                Default is 1.0.
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
                An array containing unique solution indices based on the given tolerance.

        Notes:
            This method evaluates the input solutions based on the fitness function defined
            in the OptimizationProblem class. It then compares the solutions based on their
            distances, using custom distance weights if specified.

            If 'sorted_indices' is not provided, solutions will be sorted in descending order
            of fitness values to optimize the search for unique solutions.

            If 'distance_weights' are not provided, default weights are calculated based on
            the feasible ranges of each dimension (i.e., decision variable).

            The component weights allow you to adjust the importance of each dimension when
            determining uniqueness, providing flexibility for different optimization scenarios.
        """
        # if no sorted_indices were passed evaluate the solutions to get them
        if sorted_indices is None:
            sorted_indices = np.argsort(self(solutions))[::-1]
        if len(sorted_indices) == 0 or len(solutions) == 0:
            return np.array([])
        if distance_weights is None:
            distance_weights = 1 / (self.maxs - self.mins)
        unique_solutions = [sorted_indices[0]]
        for i in range(1, len(sorted_indices)):
            distances = cdist([solutions[sorted_indices[i]]],
                              solutions[unique_solutions], w=distance_weights).flatten()
            if np.min(distances) > tolerance:
                unique_solutions.append(sorted_indices[i])
        return np.array(unique_solutions, dtype=int)

    def get_unique_solutions(self, solutions, tolerance=1.0, sorted_indices=None, distance_weights=None):
        """
        Filter and return unique solutions from a list of solutions based on a given tolerance.

        Args:
            solutions (array-like):
                List of solutions to be filtered.
            tolerance (float, optional):
                The tolerance level to consider two solutions as unique. Solutions with
                distances greater than this tolerance are considered unique.
                Default is 1.0.
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
                An array containing unique solutions based on the given tolerance.
        """
        return solutions[self.get_unique_solutions_indices(solutions, tolerance=tolerance, sorted_indices=sorted_indices, distance_weights=distance_weights)]

    def __str__(self) -> str:
        return f"""{self.name} {self.input_dimension}D"""
