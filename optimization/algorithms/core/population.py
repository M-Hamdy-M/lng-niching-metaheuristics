"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np


class Population:
    def __init__(self) -> None:
        """
        Initialize a Population object.

        Attributes:
            individuals (np.ndarray[float]): Array of individuals in the population.
            objective_values (np.ndarray): Array of objective values corresponding to the individuals.
        """
        pass

    @property
    def n_individuals(self) -> int:
        """
        Get the number of individuals in the population.

        Returns:
        int: The number of individuals.
        """
        return self.individuals.shape[0]

    @property
    def n_vars(self) -> int:
        """
        Get the number of variables for each individual.

        Returns:
            int: The number of variables.
        """
        return self.individuals.shape[1]

    @property
    def individuals(self) -> np.ndarray[float]:
        """
        Get population's individuals.

        Returns:
            np.ndarray[float]: Population's individuals.
        """
        return self.__individuals

    @individuals.setter
    def individuals(self, individuals: list[float]) -> None:
        """
        Set the individuals in the population.

        Args:
            individuals (list[float]): List of individuals to set.
        """
        self.__individuals: np.ndarray[float] = np.asarray(individuals)

    @property
    def objective_values(self):
        """
        Get the objective values associated with the population's individuals.

        Returns:
            np.ndarray: Population's individuals objective values.
        """
        return self.__objective_values

    @objective_values.setter
    def objective_values(self, objective_values: list[float]):
        """
        Set the objective values for the individuals in the population.

        Args:
            objective_values (list[float]): List of objective values to set.
        """
        self.__objective_values = np.asarray(objective_values)

    @property
    def solutions(self):
        """
        Get the solutions in the population (defaulted to individuals property).

        Returns:
            np.ndarray[float]: Array of individuals.
        """
        return self.individuals

    @property
    def solutions_objective_values(self):
        """
        Get the objective values associated with the solutions (defaulted to objective_values property).

        Returns:
            np.ndarray: Array of objective values.
        """
        return self.objective_values