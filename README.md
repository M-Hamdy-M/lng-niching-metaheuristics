# 📖 Project Overview

This repository contains the implementation of the top performing algorithms and experiments from the paper:  
**"Beyond a Single Solution: Liquefied Natural Gas Process Optimization Using Niching-Enhanced Meta-Heuristics"**  
Accepted in *Engineering Applications of Artificial Intelligence (EAAI)*.

📌 The core focus of this project is to identify **multiple high-quality solutions** for complex constrained multi-modal optimization problems, with a case study on the **DSMR process** using Aspen HYSYS®.

## 🗂️ Contents
- Implementation of the **top 5 performing meta-heuristics** on the **DSMR** process:
  - FERPSO [(Li, GECCO)](https://doi.org/10.1145/1276958.1276970)
  - r3PSO [(Li, IEEE TEVC)](http://dx.doi.org/10.1109/tevc.2009.2026270)
  - GBEST PSO [(Clerc & Kennedy, IEEE TEC)](https://doi.org/10.1109/4235.985692)
  - CLDE [(Petrowski, ICEC-96)](http://dx.doi.org/10.1109/icec.1996.542703)
  - NCDE [(Qu et al., IEEE TEVC)](http://dx.doi.org/10.1109/tevc.2011.2161873)
- Full DSMR problem setup and reproducible results
- DSMR–HYSYS® connection code for real-time simulation-based evaluation
- Modular hyperparameter tuning framework with Bayesian optimization
- Modular algorithmic interfaces
- Easy-to-run demos with visualization
- Benchmark support including CEC test suites [[CEC 2013](https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_TechnicalReport.pdf)]
---
## 🧪 Reproducing Paper Results

To replicate the results of the top-performing algorithms as presented in our [EAAI paper](https://www.sciencedirect.com/science/article/pii/S0952197625011200), use the notebooks provided under:

📁 `hysys_optimization/problems/dsmr/`  
- 🧮 [`Solving SMR.ipynb`](hysys_optimization/problems/dsmr/Solving%20SMR.ipynb): Runs optimization using the top 5 algorithms
  > ⏱️ *This notebook takes ~13 hours on an i9 processor*  
- 📊 [`Analysis.ipynb`](hysys_optimization/problems/dsmr/Analysis.ipynb): Aggregates and visualizes results

### Implemented Algorithms
Each of the top-5 performing meta-heuristics is implemented as follows:

- **FERPSO** — [`ferpso.py`](optimization/algorithms/pso/variants/ferpso.py)  
- **r3PSO** — [`lips.py`](optimization/algorithms/pso/variants/lips.py) *(local topology PSO)*  
- **GBEST PSO** — [`canonical.py`](optimization/algorithms/pso/variants/canonical.py)  
- **CLDE** — [`clde.py`](optimization/algorithms/de/variants/clde.py)  
- **NCDE** — [`nde.py`](optimization/algorithms/de/variants/nde.py)  

🧾 *Original algorithm references available in the References section*


## 🧩 DSMR Simulation

The file `Dual SMR Simulation.hsc` contains the Aspen HYSYS® simulation of the DSMR process proposed in [Qyyum et al.](https://www.sciencedirect.com/science/article/pii/S0306261920305341). To obtain the original `.hsc` file:

- 📧 Contact the authors of the original DSMR paper  [Qyyum et al.](https://www.sciencedirect.com/science/article/pii/S0306261920305341) or  
- 📬 Reach out to Abdulla Al-Saadi: **<aa1706407@qu.edu.qa>** for the version used in this publication

> ⚠️ **Note**: All HYSYS-based notebooks require **Windows OS** due to COM interface dependency

## 🔌 DSMR-HYSYS Connection Code

To bridge the optimization algorithms with Aspen HYSYS®, we use the module:  
📄 [`smr_opt_problem.py`](hysys_optimization/problems/dsmr/smr_opt_problem.py)

This file defines the `SMROptimizationProblem` class, which acts as the interface between Python and the DSMR process simulation. Here's how it works:

- During each iteration of the optimization loop, Python sends a new set of decision variables (refrigerant composition and compressor pressures) to HYSYS.
- HYSYS updates the simulation flowsheet accordingly and solves the process.
- Python waits for HYSYS to finish solving, then retrieves the **objective value** (power consumption) and **constraint values** (MITA conditions).
- These values are used to calculate the **penalized objective** (if any constraints are violated), which is returned to the optimizer.

### 💡 Features
- Uses `pywin32` to control HYSYS via COM automation
- Automatically handles simulation timeouts and retry logic
- Compatible with any algorithm through the common `ConstrainedOptimizationProblem` interface

> ⚠️ **Note**: This code requires **Windows OS** and a valid **Aspen HYSYS®** installation. Make sure the `.hsc` simulation file is available locally when running.

## 🛠️ Hyperparameter Tuning Framework

We provide a full modular framework for **automated hyperparameter tuning** of the optimization algorithms used in this study.

### 🔧 Core Tuning Logic

The tuning logic is implemented in  
📄 [`tune.py`](optimization/tuning/bo/core/tune.py)

This module leverages **Bayesian Optimization** (via [SMAC](https://automl.github.io/SMAC3/main/)) to search for the optimal hyperparameter configurations for each algorithm. It is fully generic and supports:

- Any algorithm wrapped in an `Optimizable` interface  
- Any problem derived from `OptimizationProblem`  
- Configurable number of trials and evaluation repeats  
- Persistent caching of tuned results via `.pk` files  

### 📈 DSMR-Specific Tuning

To tune on the DSMR process, use the notebook:  
📓 [`Tuning SMR.ipynb`](hysys_optimization/problems/dsmr/Tuning%20SMR.ipynb)

This notebook:
- Initializes the HYSYS-linked DSMR problem
- Initializes a list of algorithm wrappers
- Calls the `tune()` function from `tune.py` on each algorithm
- Saves and prints the best-found hyperparameters for each variant

### 🔄 Making an Algorithm Tunable

To tune or optimize an algorithm, it must be wrapped in an `OptimizableAlgorithm` interface. This wrapper:

- Specifies **which hyperparameters to tune** and their **bounds**
- Optionally sets **fixed base parameters**

Examples of such wrappers are provided for:
- **PSO variants** — [`optimizables/pso.py`](optimization/tuning/bo/optimizables/pso.py)  
- **DE variants** — [`optimizables/de.py`](optimization/tuning/bo/optimizables/de.py)

This makes it straightforward to control tuning behavior and extend support to additional algorithms.

> ⚠️ **Note**: Tuning interacts directly with the Aspen HYSYS® simulation. Ensure that the `.hsc` file is accessible and that you're running on **Windows OS** with HYSYS installed.
## ♻️ Algorithm Generalization for Custom Problems

This repository is designed with a **modular interface** to support solving **any optimization problem**.

### ➕ Add Your Own Problem

To define and solve your own problem, check the notebook:  
📄 [`Custom Problem Demo.ipynb`](demos/Custom%20Problem%20Demo.ipynb)

This notebook demonstrates:
- Running **GBEST PSO** and **r3PSO** on:
  - 🎯 Sphere (simple, convex)
  - 🔱 Himmelblau (multi-modal)
- Animated convergence plots to visualize swarm evolution

✅ Define your problem using `BenchmarkFunction` or `OptimizationProblem` from  
[`types/`](optimization/problems/types/)

```python
from optimization.problems.types import BenchmarkFunction

my_problem = BenchmarkFunction(
    name="MyTestProblem",
    bounds=[[-5, 5], [-5, 5]],
    is_maximization=False,
    evaluator=my_function,
    global_optima=[[0, 0]]
)
```

Then plug it into any optimizer:
```python
from optimization.algorithms.pso.variants import get_gbest_pso
optimizer = get_gbest_pso()
optimizer.compile(my_problem, initializer_kwargs={"n_individuals": 50})
optimizer.optimize(fes=5000)

```

---
## 💼 Software & Requirements

To run the codebase, install the following Python libraries:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `tqdm`
- `pywin32`  *(Required only for HYSYS integration on Windows)*

> ⚠️ **Note**: HYSYS-related notebooks require **Windows OS** due to COM automation with Aspen HYSYS®.


## 🔗 Benchmark Functions & References

Benchmark functions used in testing (e.g., Sphere, Himmelblau, CEC2013) are implemented in:  
📁 `optimization/problems/`

- `benchmark_functions.py`  
- `cec_benchmarks.py`  
  > 📖 Reference: [CEC2013 Technical Report](https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_TechnicalReport.pdf)

---

### 📚 Algorithm References

- **FERPSO**: X. Li, GECCO — [Link](https://doi.org/10.1145/1276958.1276970)  
- **r3PSO**: X. Li, IEEE TEVC — [Link](http://dx.doi.org/10.1109/tevc.2009.2026270)  
- **CLDE**: A. Petrowski, ICEC — [Link](http://dx.doi.org/10.1109/icec.1996.542703)  
- **NCDE**: B. Y. Qu et al., IEEE TEVC — [Link](http://dx.doi.org/10.1109/tevc.2011.2161873)  
- **GBEST PSO**: M. Clerc & J. Kennedy — [Link](https://doi.org/10.1109/4235.985692)

---

If you find this code useful in your research or projects, please cite our [paper](https://www.sciencedirect.com/science/article/pii/S0952197625011200): 
```
@article{HAMDY2025111119,
title = {Beyond a single solution: Liquefied natural gas process optimization using niching-enhanced meta-heuristics},
journal = {Engineering Applications of Artificial Intelligence},
volume = {158},
pages = {111119},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.111119},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625011200},
author = {Mohamed Hamdy and Shahd Gaben and Abdullah Al-Saadi and Abdulaziz Al-Ali and Majeda Khraisheh and Fares Almomani and Ponnuthurai N. Suganthan},
keywords = {Liquefied natural gas, Evolutionary algorithm, Niching techniques, Operational optimization, Energy consumption, Exergo-economic analysis},
}
```
---

## 📫 Contact

Thank you for your interest!  
For any queries, please don't hesitate to [contact us](mailto:mm1905748@qu.edu.qa).