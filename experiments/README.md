# GNBG-MO Experiments

This directory contains benchmarking experiments for the GNBG-II Multi-Objective extension.

## Quick Start

Run the main benchmark experiment:

```bash
python gnbg_mo_benchmark.py
```

This will test NSGA2 algorithm performance across **10-500 objectives** on GPU-accelerated GNBG-MO problems - an extreme scaling demonstration!

## Available Scripts

- `gnbg_mo_benchmark.py` - Main benchmarking script for multi-objective optimization
- `quick_test.py` - Quick test to verify GNBG-MO installation

## Configuration

Edit the configuration section in `gnbg_mo_benchmark.py` to adjust:

- Objective ranges (default: 10-100 step 10, 100-500 step 50)
- Number of runs per configuration (default: 31)
- Problem types (GF1-GF24 available - GPU Multi-Objective Functions mapping to GNBG F1-F24)
- Algorithm parameters

## Extreme Scaling Test

This benchmark tests two ranges:
- **Standard range**: 10, 20, 30, ..., 100 objectives (10 points)
- **Extreme range**: 100, 150, 200, ..., 500 objectives (9 points)

Total: **19 objective counts × 3 problems × 31 runs = 1,767 experiments per algorithm**

## Results

Results are saved to the `results/` directory:
- `gnbg_mo_benchmark_results.csv` - Detailed run data
- `summary_statistics.csv` - Aggregated statistics

## GPU Acceleration

The benchmark automatically uses GPU acceleration when available. Check the GPU column in results to verify acceleration is active.

## Selecting GNBG Functions

To test different GNBG functions, modify the problem configuration:

```python
# Test specific functions (e.g., challenging multimodal problems)
GNBG_MO_PROBLEMS = ["GF16", "GF17", "GF24"]  # Multi-component functions

# Test unimodal functions
GNBG_MO_PROBLEMS = ["GF1", "GF2", "GF3"]     # Well-conditioned to ill-conditioned

# Test single-component multimodal
GNBG_MO_PROBLEMS = ["GF7", "GF9", "GF15"]    # Various multimodal landscapes
```

See the main README for complete GF1-GF24 function characteristics and mapping to GNBG F1-F24.

## Adding More Algorithms

To test NSGA3, uncomment the relevant lines in the configuration:

```python
# ALGORITHMS = ["NSGA2", "NSGA3"]  # Uncomment to include NSGA3
```

And uncomment the `create_nsga3_algorithm` function.