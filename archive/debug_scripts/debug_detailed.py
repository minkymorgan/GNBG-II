#!/usr/bin/env python3
"""
Debug with detailed logging enabled
"""

import logging
import numpy as np
import gnbg_gpu.gnbg_gpu as rust_module

# Enable Rust logging
import os
os.environ['RUST_LOG'] = 'debug'

print("🔍 Detailed Debugging with Rust Logging")
print("=" * 50)

pymoo_interface = rust_module.pymoo_interface

# Create problem that we know causes NaN
config = {'n_var': 5, 'n_obj': 2}
problem = pymoo_interface.create_gnbg_problem(config, "debug_detailed")
problem.set_gpu_enabled(False)  # Use CPU for easier debugging

print(f"Created problem with {config}")
stats = problem.get_stats()
print(f"Position vars: {stats['n_position_vars']}, Distance vars: {stats['n_distance_vars']}")

# Test the problematic solution
solution = [-100.0, -100.0, -100.0, -100.0, -100.0]
print(f"\nTesting solution: {solution}")

try:
    objectives = problem.evaluate_single(solution)
    finite = np.all(np.isfinite(objectives))
    print(f"Result: {objectives} (finite: {finite})")
except Exception as e:
    print(f"ERROR: {e}")

print("\n🎯 Check logs above for Rust debug output")