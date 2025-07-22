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