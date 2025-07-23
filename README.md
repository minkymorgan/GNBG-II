# GNBG-II

Generalized Numerical Benchmark Generator - II: A benchmark suite consisting of 24 test problems for the 2025 CEC and GECCO competitions.

This repository contains a GPU-accelerated implementation using Rust and WebGPU, with Python bindings. Performance measurements indicate 6.8x to 12.1x speedup over CPU implementations while maintaining numerical consistency.

## Requirements

- Rust (stable)
- Python 3.11+
- Maturin

## Installation

```bash
git clone https://github.com/minkymorgan/GNBG-II.git
cd GNBG-II
pip install maturin
maturin develop --features python
```

## Usage

```python
import gnbg_gpu
import numpy as np

# Problem F1
gnbg = gnbg_gpu.GNBGGpu(1, use_gpu=True)

# Single evaluation
solution = np.random.uniform(-100, 100, gnbg.dimension)
fitness = gnbg.fitness_single(solution)

# Batch evaluation
solutions = np.random.uniform(-100, 100, (100, gnbg.dimension))
fitness_batch = gnbg.fitness(solutions)
```

## API

### GNBGGpu(problem_index, use_gpu=True)

Parameters:
- problem_index: Problem number (1-24)
- use_gpu: Enable GPU acceleration

Methods:
- fitness(X): Batch evaluation
- fitness_single(x): Single evaluation
- reset(): Reset counters

### GNBG(problem_index, use_gpu=True)

Python wrapper maintaining compatibility with original interface.

## GNBG Function Reference (F1-F24 to GF Mapping)

The GNBG-II suite contains 24 carefully designed optimization problems. For multi-objective extensions, we use GF (GPU Functions) terminology:

### Unimodal Functions (GF1-GF6)
| GF | GNBG | Characteristics | Optimum | Max Evals |
|----|------|----------------|---------|-----------|
| GF1 | F1 | Well-conditioned, separable | -1081.984 | 500K |
| GF2 | F2 | Moderately conditioned | -104.865 | 500K |
| GF3 | F3 | Ill-conditioned | -51.479 | 500K |
| GF4 | F4 | Non-separable, rotated | -930.063 | 500K |
| GF5 | F5 | Non-separable, shifted | -525.477 | 500K |
| GF6 | F6 | Non-separable, rotated+shifted | -337.509 | 500K |

### Single-Component Multimodal (GF7-GF15)
| GF | GNBG | Characteristics | Optimum | Max Evals |
|----|------|----------------|---------|-----------|
| GF7 | F7 | Many local optima | -874.478 | 500K |
| GF8 | F8 | Deceptive landscape | -318.801 | 500K |
| GF9 | F9 | Highly multimodal | -1030.301 | 500K |
| GF10 | F10 | Rugged surface | -884.736 | 500K |
| GF11 | F11 | Asymmetric basins | -843.147 | 500K |
| GF12 | F12 | Variable interactions | -401.147 | 500K |
| GF13 | F13 | Non-linear basins | -159.612 | 500K |
| GF14 | F14 | Conditioning+multimodal | -189.941 | 500K |
| GF15 | F15 | Complex landscape | -624.056 | 500K |

### Multi-Component Multimodal (GF16-GF24)
| GF | GNBG | Components | Characteristics | Optimum | Max Evals |
|----|------|------------|----------------|---------|-----------|
| GF16 | F16 | 5 | Hybrid composition | -4208.486 | 1M |
| GF17 | F17 | 5 | Rotated composition | -4381.528 | 1M |
| GF18 | F18 | 5 | Non-separable hybrid | -4046.060 | 1M |
| GF19 | F19 | 5 | Complex interactions | -966.475 | 1M |
| GF20 | F20 | 2 | Dual-component | -1000.000 | 1M |
| GF21 | F21 | 5 | Advanced hybrid | -992.221 | 1M |
| GF22 | F22 | 5 | Extreme conditioning | -99.363 | 1M |
| GF23 | F23 | 5 | Maximum complexity | -99.388 | 1M |
| GF24 | F24 | 7 | Ultimate challenge | -99.917 | 1M |

### Running Specific Functions

To run any of the 24 GNBG functions in experiments:

```python
# Single-objective optimization
gnbg = gnbg_gpu.GNBGGpu(problem_index=5)  # For F5/GF5
solutions = np.random.uniform(-100, 100, (100, 30))
fitness_values = gnbg.fitness(solutions)

# Multi-objective GNBG-MO (experiments/)
from gnbg_gpu.multi_objective import GNBGMultiObjectiveProblem
problem = GNBGMultiObjectiveProblem.from_gnbg_function(
    gnbg_function=5,  # F5/GF5
    n_obj=10,         # 10 objectives
    n_var=30          # 30 variables
)
```

**Note**: All functions use 30-dimensional search space [-100, 100]^30 with acceptance threshold 1e-08.

## Performance

| Objectives | GPU (eval/s) | CPU (eval/s) | Speedup |
|------------|--------------|--------------|----------|
| 5          | 206,558      | 30,355       | 6.8x     |
| 10         | 408,057      | 60,786       | 6.7x     |
| 20         | 989,193      | 122,111      | 8.1x     |
| 30         | 2,196,089    | 181,536      | 12.1x    |

## License

GNU General Public License v3.0

Original GNBG-II Authors:
- Danial Yazdani (danial.yazdani@gmail.com) - Copyright (c) 2023
- Vladimir Stanovov (vladimirstanovov@yandex.ru) - C++ Implementation (2024)

GPU Acceleration:
- Andrew Morgan (minkymorgan@gmail.com) - Rust GPU Implementation (2025)