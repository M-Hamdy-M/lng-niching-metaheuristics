"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from optimization.algorithms.core import Population


class Swarm(Population):
    """Class representing a swarm of individuals in Particle Swarm Optimization (PSO).

    This class extends the Population class and adds specific attributes and methods relevant to swarm-based optimization algorithms.
    """

    def __init__(self, n_individuals: int, n_vars: int, n_obj: int = 1) -> None:
        """Initialize the Swarm with the number of individuals, variables, and objective functions.

        Args:
            n_individuals (int): Number of individuals (particles) in the swarm.
            n_vars (int): Number of variables in each individual's representation.
            n_obj (int, optional): Number of objective functions. Defaults to 1.

        """
        super().__init__()
        self.individuals = np.empty((n_individuals, n_vars))

    @property
    def particles_velocities(self) -> np.ndarray[float]:
        """Get the velocities of particles in the swarm.

        Returns:
            np.ndarray[float]: Velocities of particles.

        """
        return self.__particles_velocities

    @particles_velocities.setter
    def particles_velocities(self, particles_velocities: list[float]) -> None:
        """Set the velocities of particles in the swarm.

        Args:
            particles_velocities (list[float]): Velocities of particles.

        Raises:
            ValueError: If the shape of particles' velocities is not as expected.

        """
        tmp: np.ndarray[float] = np.array(particles_velocities)
        if tmp.shape != (self.n_individuals, self.n_vars):
            raise ValueError(
                f"Invalid shape of particles' velocities. Expected {(self.n_individuals, self.n_vars)}, but got {tmp.shape}")
        self.__particles_velocities: np.ndarray[float] = tmp

    @property
    def objective_values(self):
        """Get the objective function values of the individuals.

        Returns:
            np.ndarray: Objective function values of the individuals.

        """
        return super().objective_values

    @objective_values.setter
    def objective_values(self, objective_values: np.ndarray):
        """Set the objective function values of the individuals and update personal bests.

        Args:
            objective_values (np.ndarray): Objective function values of the individuals.

        Raises:
            ValueError: If the shape of personal bests or personal best fitnesses is not as expected.

        """
        super(Swarm, self.__class__).objective_values.fset(
            self, objective_values)
        if not hasattr(self, "personal_bests"):
            self.personal_bests = self.individuals
            self.personal_bests_fitnesses = self.objective_values
        else:
            mask = self.objective_values >= self.personal_bests_fitnesses
            self.personal_bests[mask] = self.individuals[mask].copy()
            self.personal_bests_fitnesses[mask] = self.objective_values[mask].copy(
            )

    @property
    def personal_bests(self) -> np.ndarray[float]:
        """Get the personal best positions of individuals.

        Returns:
            np.ndarray[float]: Personal best positions of individuals.

        """
        return self.__personal_bests

    @personal_bests.setter
    def personal_bests(self, personal_bests: list[float]) -> None:
        """Set the personal best positions of individuals.

        Args:
            personal_bests (list[float]): Personal best positions of individuals.

        Raises:
            ValueError: If the shape of personal bests is not as expected.

        """
        tmp: np.ndarray[float] = np.array(personal_bests)
        if tmp.shape != (self.n_individuals, self.n_vars):
            raise ValueError(
                f"Invalid Shape for personal_bests: Expected {(self.n_individuals, self.n_vars)}, but got {tmp.shape}")
        self.__personal_bests: np.ndarray[float] = tmp

    @property
    def personal_bests_fitnesses(self) -> np.ndarray[float]:
        """Get the fitness values of personal best positions.

        Returns:
            np.ndarray[float]: Fitness values of personal best positions.

        """
        return self.__personal_bests_fitnesses

    @personal_bests_fitnesses.setter
    def personal_bests_fitnesses(self, fitnesses: list[float]) -> None:
        """Set the fitness values of personal best positions.

        Args:
            fitnesses (list[float]): Fitness values of personal best positions.

        Raises:
            ValueError: If the shape of personal best fitnesses is not as expected.

        """
        tmp: np.ndarray[float] = np.array(fitnesses)
        if tmp.shape != (self.n_individuals, ):
            raise ValueError(
                f"Invalid Shape for personal_bests_fitnesses: Expected {(self.n_individuals, )}, but got {tmp.shape}")
        self.__personal_bests_fitnesses: np.ndarray[float] = tmp

    @property
    def global_bests(self) -> np.ndarray[float]:
        """Get the global best positions of the swarm.

        Returns:
            np.ndarray[float]: Global best positions of the swarm.

        """
        return self.__global_bests

    @global_bests.setter
    def global_bests(self, global_bests: list[int]) -> None:
        """Set the global best positions of the swarm.

        Args:
            global_bests (list[int]): Global best positions of the swarm.

        Raises:
            ValueError: If the shape of global bests is not as expected.

        """
        tmp: np.ndarray[float] = np.array(global_bests, dtype=int)
        if 1 <= tmp.ndim <= 2 and tmp.shape[0] == self.n_individuals:
            self.__global_bests: np.ndarray[float] = tmp
        else:
            raise ValueError(
                f"Invalid Shape for global_bests: Expected {(self.n_individuals, )}, but got {tmp.shape}")

    @property
    def solutions(self):
        """Alias for personal best positions.

        Returns:
            np.ndarray[float]: Personal best positions of individuals.

        """
        return self.personal_bests

    @property
    def solutions_objective_values(self):
        """Alias for objective values of personal best positions.

        Returns:
            np.ndarray[float]: Objective function values of personal best positions.

        """
        return self.personal_bests_fitnesses
