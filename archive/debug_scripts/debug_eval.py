#!/usr/bin/env python3
"""
Simple debugging script to isolate the NaN issue
"""

import numpy as np
import gnbg_gpu.gnbg_gpu as rust_module

print("🔍 Debugging GNBG Multi-Objective Evaluation")
print("=" * 50)

# Test 1: Create problem directly from Rust
pymoo_interface = rust_module.pymoo_interface

config = {
    'n_var': 5, 
    'n_obj': 2,
    'wfg': {'problem': 1, 'n_obj': 2}
}

problem = pymoo_interface.create_gnbg_problem(config, "debug_test")
print(f"Created problem: {problem.name}")
print(f"Variables: {problem.n_var}, Objectives: {problem.n_obj}")
print(f"GPU enabled: {problem.is_gpu_enabled()}")

# Test 2: Try different inputs
test_solutions = [
    [0.0, 0.0, 0.0, 0.0, 0.0],        # All zeros
    [1.0, 1.0, 1.0, 1.0, 1.0],        # All ones
    [50.0, 50.0, 50.0, 50.0, 50.0],   # Middle range
    [-50.0, -50.0, -50.0, -50.0, -50.0], # Negative middle
    [100.0, 100.0, 100.0, 100.0, 100.0], # Upper bound
    [-100.0, -100.0, -100.0, -100.0, -100.0], # Lower bound
]

print(f"\n📋 Testing single solution evaluation:")
for i, solution in enumerate(test_solutions):
    try:
        objectives = problem.evaluate_single(solution)
        finite = np.all(np.isfinite(objectives))
        print(f"  Solution {i}: {solution[:2]}... → {objectives} (finite: {finite})")
    except Exception as e:
        print(f"  Solution {i}: {solution[:2]}... → ERROR: {e}")

# Test 3: Disable GPU and try CPU path
print(f"\n📋 Testing with CPU only:")
problem.set_gpu_enabled(False)
print(f"GPU disabled: {not problem.is_gpu_enabled()}")

for i, solution in enumerate(test_solutions[:3]):  # Test first 3
    try:
        objectives = problem.evaluate_single(solution)
        finite = np.all(np.isfinite(objectives))
        print(f"  CPU Solution {i}: {solution[:2]}... → {objectives} (finite: {finite})")
    except Exception as e:
        print(f"  CPU Solution {i}: {solution[:2]}... → ERROR: {e}")

# Test 4: Check problem stats
print(f"\n📋 Problem Statistics:")
try:
    stats = problem.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"  ERROR getting stats: {e}")

print(f"\n🎯 Debugging Complete")