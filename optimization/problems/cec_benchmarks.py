"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy & Shahd Gaben                    ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import os
import numpy as np
import typing
from .benchmark_functions import evaluate_rosenbrock, evaluate_sphere, evaluate_rastrigin
from .types import BenchmarkFunction, CompositeBenchmarkFunction
# The following are some basic and composite niching benchmark functions that were put together in
# the technical report titled [Benchmark Functions for CEC’2013 Special Session and Competition on Niching Methods for Multimodal Function Optimization]

module_path = os.path.abspath(os.path.dirname(__file__))


# Func 1 - Five-Uneven-Peak Trap 1D

def evaluate_five_uneven_peak_trap1d(X):
    if np.max(X) > 30 or np.min(X) < 0:
        raise ValueError(
            "Invalid value passed to function, all X values must be between 0 and 30")
    tmp = X.flatten()
    Y = np.empty((X.shape[0]))

    mask = (tmp < 2.5) * (tmp >= 0)
    Y[mask] = 80 * (2.5 - tmp[mask])

    mask = (tmp < 5) * (tmp >= 2.5)
    Y[mask] = 64 * (tmp[mask] - 2.5)

    mask = (tmp < 7.5) * (tmp >= 5.0)
    Y[mask] = 64 * (7.5 - tmp[mask])

    mask = (tmp < 12.5) * (tmp >= 7.5)
    Y[mask] = 28 * (tmp[mask] - 7.5)

    mask = (tmp < 17.5) * (tmp >= 12.5)
    Y[mask] = 28 * (17.5 - tmp[mask])

    mask = (tmp < 22.5) * (tmp >= 17.5)
    Y[mask] = 32 * (tmp[mask] - 17.5)

    mask = (tmp < 27.5) * (tmp >= 22.5)
    Y[mask] = 32 * (27.5 - tmp[mask])

    mask = (tmp <= 30) * (tmp >= 27.5)
    Y[mask] = 80 * (tmp[mask] - 27.5)
    return Y


five_uneven_peak_trap1d_benchmark = BenchmarkFunction(
    name="Five-Uneven-Peak Trap",
    evaluator=evaluate_five_uneven_peak_trap1d,
    bounds=[[0, 30]],
    global_optima=[
        [0],
        [30],
    ],
    is_maximization=True,
    niche_radius=0.01,
    max_fitness_evaluations=50000,
)
# Func 2 - Equal Maxima 1D


def evaluate_equal_maxima1d(X):
    return np.power(np.sin(5 * np.pi * X), 6).flatten()


equal_maxima1d_benchmark = BenchmarkFunction(
    name="Equal Maxima",
    evaluator=evaluate_equal_maxima1d,
    bounds=[[0, 1]],
    global_optima=[
        [0.1],
        [0.3],
        [0.5],
        [0.7],
        [0.9],
    ],
    is_maximization=True,
    niche_radius=0.01,
    max_fitness_evaluations=50000,
)
# Func 3 - Uneven Decreasing Maxima 1D


def evaluate_uneven_decreasing_maxima1d(X):
    return np.exp(-2 * np.log(2) * np.square((X - 0.08) / 0.854)).flatten() * evaluate_equal_maxima1d((np.power(X, 3/4) - 0.05))


uneven_decreasing_maxima1d_benchmark = BenchmarkFunction(
    name="Uneven Decreasing Maxima",
    evaluator=evaluate_uneven_decreasing_maxima1d,
    bounds=[[0, 1]],
    global_optima=[
        [0.079699779582100],
    ],
    is_maximization=True,
    niche_radius=0.01,
    max_fitness_evaluations=50000,
)
# Func 4 - Inverted Himmelblau 2D


def evaluate_inverted_himmelblau2d(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return 200 - (((x1 ** 2) + x2 - 11) ** 2) - ((x1 + (x2 ** 2) - 7) ** 2)


inverted_himmelblau2d_benchmark = BenchmarkFunction(
    name="Himmelblau",
    evaluator=evaluate_inverted_himmelblau2d,
    bounds=[[-6, 6]] * 2,
    global_optima=[
        [3, 2],
        [-2.805118094822989, 3.131312538494919],
        [-3.779310265963066, -3.283185984612214],
        [3.584428351760445, -1.848126540197251]
    ],
    is_maximization=True,
    niche_radius=0.01,
    max_fitness_evaluations=50000,
)


# Func 5 - Six-Hump Camel Back 2D
def evaluate_six_hump_camel_back2d(X):
    x1_2 = np.square(X[:, 0])
    x2_2 = np.square(X[:, 1])
    return (
        x1_2 * (4 - 2.1 * x1_2 + (np.square(x1_2)) / 3)
        + (np.prod(X, axis=-1))
        + x2_2 * (-4 + 4 * x2_2)
    )


six_humb_camel2d_benchmark = BenchmarkFunction(
    name="Six-Hump Camel Back",
    evaluator=evaluate_six_hump_camel_back2d,
    bounds=[[-1.9, 1.9], [-1.1, 1.1]],
    global_optima=[
        [0.089842008935272,  -0.712656403019058],
        [-0.089842008935272,  0.712656403019058]
    ],
    is_maximization=False,
    niche_radius=0.5,
    max_fitness_evaluations=50000,
)

# Func 6 & 7 - Inverted Shubert 2D & 3D


def evaluate_inverted_shubert(X):
    Y = np.zeros(X.shape)
    for i in range(1, 6):
        Y += i * np.cos((i + 1) * X + i)
    return np.prod(Y, axis=-1)


inverted_shubert2d_benchmark = BenchmarkFunction(
    name="Inverted Shubert",
    evaluator=evaluate_inverted_shubert,
    bounds=[[-10, 10]] * 2,
    global_optima=np.load(os.path.join(
        module_path, "data/cec_inverted_shubert2d_optima.npy")),
    is_maximization=False,
    niche_radius=0.5,
    max_fitness_evaluations=200000,
)
inverted_shubert3d_benchmark = BenchmarkFunction(
    name="Inverted Shubert",
    evaluator=evaluate_inverted_shubert,
    bounds=[[-10, 10]] * 3,
    global_optima=np.load(os.path.join(
        module_path, "data/cec_inverted_shubert3d_optima.npy")),
    is_maximization=False,
    niche_radius=0.2,
    max_fitness_evaluations=200000,
)

# Func 8 & 9 - Inverted Vincent 2D & 3D


def evaluate_inverted_vincent(X):
    return np.sum(np.sin(10 * np.log(X)), axis=-1) / X.shape[1]


inverted_vincent2d_benchmark = BenchmarkFunction(
    name="Inverted Vincent",
    evaluator=evaluate_inverted_vincent,
    bounds=[[0.25, 10]] * 2,
    global_optima=np.load(os.path.join(
        module_path, "data/cec_inverted_vincent2d_optima.npy")),
    is_maximization=True,
    niche_radius=0.01,
    max_fitness_evaluations=200000,
)
inverted_vincent3d_benchmark = BenchmarkFunction(
    name="Inverted Vincent",
    evaluator=evaluate_inverted_vincent,
    bounds=[[0.25, 10]] * 3,
    global_optima=np.load(os.path.join(
        module_path, "data/cec_inverted_vincent3d_optima.npy")),
    is_maximization=True,
    niche_radius=0.01,
    max_fitness_evaluations=400000,
)
# Func 10 - Modified Rastrigin 2D


def evaluate_modified_rastrigin(X, ks):
    return np.sum(10 + 9 * np.cos(2 * np.pi * ks * X), axis=-1)


def evaluate_cec_modified_rastrigin2d(X):
    ks = np.array([3, 4])
    return evaluate_modified_rastrigin(X, ks)


modified_rastrigin2d_benchmark = BenchmarkFunction(
    name="Modified Rastrigin",
    evaluator=evaluate_cec_modified_rastrigin2d,
    bounds=[[0, 1]] * 2,
    global_optima=np.load(os.path.join(
        module_path, "data/cec_modified_rastrigin2d_optima.npy")),
    is_maximization=False,
    niche_radius=0.01,
    max_fitness_evaluations=200000,
)

# Basic Functions


def evaluate_grienwank(X):
    return (np.sum(np.square(X), axis=-1) / 4000) - np.prod(np.cos(X / np.sqrt(np.arange(1, X.shape[1] + 1))), axis=-1) + 1


def evaluate_weierstrass(X, kmax=20, alpha=0.5, beta=3):
    kmax = 20
    alpha = 0.5
    beta = 3
    f1 = np.zeros(X.shape)
    f2 = 0
    for k in np.arange(kmax+1):
        f1 += np.power(alpha, k) * np.cos(2 * np.pi *
                                          np.power(beta, k) * (X + 0.5))
        f2 += np.power(alpha, k) * np.cos(2 * np.pi * np.power(beta, k) * 0.5)
    return np.sum(f1, axis=-1) - X.shape[1] * f2


def evaluate_expanded_griewank_rosenbrock(X):
    Y = np.zeros((X.shape[0], ))
    for i in range(0, X.shape[1] - 1):
        # GitHub Implementation
        Y += evaluate_grienwank(np.expand_dims(
            evaluate_rosenbrock(X[:, [i, i+1]] + 1), axis=-1))
        # Paper Implementation
        # Y += evaluate_grienwank(np.expand_dims(
        #     evaluate_rosenbrock(X[:, [i, i+1]] ), axis=-1))
    # GitHub Implementation
    Y += evaluate_grienwank(np.expand_dims(
        evaluate_rosenbrock(X[:, [X.shape[1] - 1, 0]] + 1), axis=-1))
    # Paper Implementation
    # Y += evaluate_grienwank(np.expand_dims(
    #     evaluate_rosenbrock(X[:, [X.shape[1] - 1, 0]] + 1), axis=-1))
    return Y

# Composite Functions


def get_multi_dimensional_compositebf_getter(
        bounds_getter: typing.Callable[[int], np.ndarray[float]],
        global_optima_getter: typing.Callable[[int], np.ndarray[float]],
        local_optima_getter: typing.Optional[typing.Callable[[
            int], np.ndarray[float]]] = None,
    ms_getter: typing.Optional[typing.Callable[[
        int], np.ndarray[float]]] = None,
        **benchmark_function_args) -> typing.Callable[[int], CompositeBenchmarkFunction]:
    def func(input_dimension):
        return CompositeBenchmarkFunction(
            **benchmark_function_args,
            bounds=bounds_getter(input_dimension),
            global_optima=global_optima_getter(input_dimension),
            local_optima=local_optima_getter(
                input_dimension) if local_optima_getter else None,
            ms=ms_getter(input_dimension) if ms_getter else None
        )
    return func


path = os.path.join(module_path, "data/optima.npy")


o = np.load(path)

# Func 11
# CF1
fs1 = [
    evaluate_grienwank,
    evaluate_grienwank,
    evaluate_weierstrass,
    evaluate_weierstrass,
    evaluate_sphere,
    evaluate_sphere,
]
lambdas = np.array([1.0, 1.0, 8.0, 8.0, 1.0 / 5.0, 1.0 / 5.0])


def cf1_bounds_getter(d):
    return np.full((d, 2), [-5., 5.])


def cf1_go_getter(d):
    return o[:len(fs1), :d]


def get_CF1(input_dimension) -> CompositeBenchmarkFunction:
    return get_multi_dimensional_compositebf_getter(
        bounds_getter=cf1_bounds_getter,
        global_optima_getter=cf1_go_getter,
        name="CEC CF1",
        is_maximization=False,
        niche_radius=0.01,
        basic_functions=fs1,
        lambdas=lambdas,
    )(input_dimension)


CF12d: CompositeBenchmarkFunction = get_CF1(input_dimension=2)
CF12d.max_fitness_evaluations = 200000

# Func 12
# CF2
lambdas = np.array(
    [1.0, 1.0, 10.0, 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 7.0, 1.0 / 7.0]
)
fs2 = [
    evaluate_rastrigin,
    evaluate_rastrigin,
    evaluate_weierstrass,
    evaluate_weierstrass,
    evaluate_grienwank,
    evaluate_grienwank,
    evaluate_sphere,
    evaluate_sphere,
]


def cf2_bounds_getter(d):
    return np.full((d, 2), [-5., 5.])


def cf2_go_getter(d):
    return o[:len(fs2), :d]


def get_CF2(input_dimension) -> CompositeBenchmarkFunction:
    return get_multi_dimensional_compositebf_getter(
        bounds_getter=cf2_bounds_getter,
        global_optima_getter=cf2_go_getter,
        name="CEC CF2",
        is_maximization=False,
        niche_radius=0.01,
        basic_functions=fs2,
        lambdas=lambdas,
    )(input_dimension)


CF22d = get_CF2(input_dimension=2)
CF22d.max_fitness_evaluations = 200000


def load_rotation_mat(file_name, n_functions):
    return np.load(file_name)[:n_functions]


# Func 13 & 14 & 15 & 16
# CF3
fs3 = [
    evaluate_expanded_griewank_rosenbrock,
    evaluate_expanded_griewank_rosenbrock,
    evaluate_weierstrass,
    evaluate_weierstrass,
    evaluate_grienwank,
    evaluate_grienwank,
]


def cf3_bounds_getter(d):
    return np.full((d, 2), [-5., 5.])


def cf3_go_getter(d):
    return o[:len(fs3), :d]


def cf3_ms_getter(d):
    return load_rotation_mat(
        os.path.join(module_path, f"data/CF3_M_D{str(d)}.npy"), len(fs3)) if d in [2, 3, 5, 10, 20] else None


def get_CF3(input_dimension) -> CompositeBenchmarkFunction:
    return get_multi_dimensional_compositebf_getter(
        bounds_getter=cf3_bounds_getter,
        global_optima_getter=cf3_go_getter,
        name="CEC CF3",
        is_maximization=False,
        niche_radius=0.01,
        basic_functions=fs3,
        lambdas=np.array([1.0 / 4.0, 1.0 / 10.0, 2.0, 1.0, 2.0, 5.0]),
        sigmas=np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
        ms_getter=cf3_ms_getter
    )(input_dimension)


CF32d: CompositeBenchmarkFunction = get_CF3(input_dimension=2)
CF32d.max_fitness_evaluations = 200000

CF33d: CompositeBenchmarkFunction = get_CF3(input_dimension=3)
CF33d.max_fitness_evaluations = 400000

CF35d: CompositeBenchmarkFunction = get_CF3(input_dimension=5)
CF35d.max_fitness_evaluations = 400000

CF310d: CompositeBenchmarkFunction = get_CF3(input_dimension=10)
CF310d.max_fitness_evaluations = 400000


# Func 17 & 18 & 19 & 20
# CF4
fs4 = [
    evaluate_rastrigin,
    evaluate_rastrigin,
    evaluate_expanded_griewank_rosenbrock,
    evaluate_expanded_griewank_rosenbrock,
    evaluate_weierstrass,
    evaluate_weierstrass,
    evaluate_grienwank,
    evaluate_grienwank,
]
sigmas = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def cf4_bounds_getter(d):
    return np.full((d, 2), [-5., 5.])


def cf4_go_getter(d):
    return o[:len(fs4), :d]


def cf4_ms_getter(d):
    return load_rotation_mat(
        os.path.join(module_path, f"data/CF4_M_D{str(d)}.npy"), len(fs4)) if d in [2, 3, 5, 10, 20] else None


def get_CF4(input_dimension) -> CompositeBenchmarkFunction:
    return get_multi_dimensional_compositebf_getter(
        bounds_getter=cf4_bounds_getter,
        global_optima_getter=cf4_go_getter,
        name="CEC CF4",
        is_maximization=False,
        niche_radius=0.01,
        basic_functions=fs4,
        lambdas=np.array([4.0, 1.0, 4.0, 1.0, 1.0 / 10.0,
                         1.0 / 5.0, 1.0 / 10.0, 1.0 / 40.0]),
        sigmas=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        ms_getter=cf4_ms_getter
    )(input_dimension)


CF43d: CompositeBenchmarkFunction = get_CF4(input_dimension=3)
CF43d.max_fitness_evaluations = 400000

CF45d: CompositeBenchmarkFunction = get_CF4(input_dimension=5)
CF45d.max_fitness_evaluations = 400000

CF410d: CompositeBenchmarkFunction = get_CF4(input_dimension=10)
CF410d.max_fitness_evaluations = 400000

CF420d: CompositeBenchmarkFunction = get_CF4(input_dimension=20)
CF420d.max_fitness_evaluations = 400000


cec_suite: list[BenchmarkFunction] = [
    five_uneven_peak_trap1d_benchmark,
    equal_maxima1d_benchmark,
    uneven_decreasing_maxima1d_benchmark,
    inverted_himmelblau2d_benchmark,
    six_humb_camel2d_benchmark,
    inverted_shubert2d_benchmark,
    inverted_shubert3d_benchmark,
    inverted_vincent2d_benchmark,
    inverted_vincent3d_benchmark,
    modified_rastrigin2d_benchmark,
    CF12d,
    CF22d,
    CF32d,
    CF33d,
    CF35d,
    CF310d,
    CF43d,
    CF45d,
    CF410d,
    CF420d,
]
