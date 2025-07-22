# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Development Principle

NEVER hide bugs, provide failover functions, or silently overlook a failure. Raise an Error, halt processing, report on the issue. The job is to find and fix bugs and root causes, not hide them or obfuscate them, but to surface bugs.

## Project Overview

GNBG-II (Generalized Numerical Benchmark Generator - II) is a numerical optimization benchmark suite for the 2025 CEC and GECCO competitions. It contains 24 challenging optimization problems (f1-f24) with implementations in Python, MATLAB, C++, and Rust (GPU-accelerated).

## Key Commands

### Python Implementation
```bash
# Extract if needed
unzip GNBG_Instances.Python-main.zip

# Install dependencies
pip install numpy scipy matplotlib

# Run benchmark with example optimizer
cd Python_Implementation/GNBG_Instances.Python-main
python GNBG_instances.py
```

### MATLAB Implementation
```bash
# Extract if needed
unzip "GNBG II- Instance.MATLAB.zip"

# Run in MATLAB
cd MATLAB_Implementation/GNBG\ II-\ Instance.MATLAB
matlab -r "run('main.m')"
```

### C++ Implementation
```bash
# Extract if needed
unzip GNBG-Instance-C-main.zip

# Convert data files (required)
cd C_Implementation/GNBG-Instance-C-main
python convert.py

# Compile and run
g++ -O3 gnbg-c++.cpp -o gnbg
./gnbg
```

### Rust GPU Implementation
```bash
# Build and run
cargo build --release
cargo run --release
```

## Architecture Overview

### Core Components

1. **Benchmark Functions**: 24 test problems with varying characteristics:
   - Stored as `.mat` files containing rotation matrices, shifts, and parameters
   - Each function has specific dimension, evaluation budget, and acceptance threshold
   - Search space: [-100, 100] per dimension
   - Characteristics include ruggedness, conditioning, symmetry, variable interactions, modality, deceptiveness, and basin linearity

2. **Language Implementations**:
   - **Python**: Object-oriented with GNBG class, uses SciPy's differential_evolution
   - **MATLAB**: Modular with separate fitness.m, includes custom DE/rand/1/bin
   - **C++**: High-performance with GNBG and Optimizer classes, batch processes all 24 functions
   - **Rust**: GPU-accelerated using wgpu compute shaders, supports both CPU and GPU evaluation

3. **Fitness Evaluation Pipeline**:
   - Input transformation (shift, rotation)
   - Multiple component evaluations (BasicFunc1-6)
   - Non-linear transformation function
   - Weighted combination with bias terms
   - Evaluation counting and acceptance tracking

### Key Classes/Functions

- `GNBG` (Python/C++): Main benchmark class
  - `fitness(X)`: Evaluates solution(s), handles batch evaluation
  - `transform(X, Alpha, Beta)`: Non-linear transformation
- `fitness()` (MATLAB): Core evaluation function
- `CPUEvaluator`/`GpuExecutor` (Rust): Compute backends
- Example optimizers: Differential Evolution implementations

### Integration Points

When implementing new optimizers:
1. Create solutions within bounds [-100, 100]
2. Call fitness function with solution matrix/vector
3. Track evaluations against MaxEvals budget
4. Monitor AcceptanceReachPoint for convergence
5. Store BestFoundResult and FEhistory for analysis

## Development Workflow

1. **Choose Implementation Language**:
   - Python: Rapid prototyping, ML library integration
   - MATLAB: Optimization toolbox integration
   - C++: Maximum single-thread performance
   - Rust: GPU acceleration for parallel evaluation

2. **Load Problem Instance**:
   ```python
   # Python example
   ProblemIndex = 5  # Select f5
   # Parameters auto-loaded from f5.mat
   ```

3. **Implement Optimizer**:
   ```python
   # Use fitness interface
   fitness_values = gnbg.fitness(population)
   # Check acceptance
   if gnbg.AcceptanceReachPoint < np.inf:
       print("Target reached!")
   ```

4. **Analyze Results**:
   - BestFoundResult: Best fitness found
   - AcceptanceReachPoint: FE when threshold reached
   - FEhistory: Full convergence curve
   - Error: abs(BestFoundResult - OptimumValue)

## Testing New Algorithms

No formal test suite exists. Validation approach:
1. Run on simple problems (f1-f5) first
2. Compare against provided DE baseline:
   - Python: scipy.optimize.differential_evolution
   - MATLAB/C++: Custom DE/rand/1/bin
3. Verify evaluation counting matches expected behavior
4. Check boundary handling for solutions outside [-100, 100]
5. Run multiple seeds (31 runs standard for competition)

## Important Implementation Notes

- All .mat files must be in same directory as implementation
- C++ requires converting .mat to .txt via convert.py
- Fitness includes complex transformations - no direct gradients
- Problems designed to exploit specific algorithm weaknesses
- Winner algorithms in `Winner_Algorithms/` for reference
- Rust GPU uses f32 for compute, f64 for CPU validation
- MATLAB uses midpoint-target boundary repair strategy