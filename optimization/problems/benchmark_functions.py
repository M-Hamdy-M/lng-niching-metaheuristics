"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy & Shahd Gaben                    ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import numpy as np

def evaluate_rastrigin(X):
    return 10 * X.shape[1] + np.sum(np.square(X) - 10 * np.cos(2 * np.pi * X), axis=1)

def evaluate_sphere(X):
    return np.sum(np.square(X), axis=1)

def evaluate_rosenbrock(X):
    return np.sum(100 * np.square(X[:, 1:] - np.square(X[:, :-1])) + np.square(X[:, :-1] - 1), 1)