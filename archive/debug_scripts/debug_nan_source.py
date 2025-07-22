#!/usr/bin/env python3
"""Find the exact source of the NaN values"""

import numpy as np
import gnbg_gpu.gnbg_gpu as rust_module
import os

# Enable debug logging
os.environ['RUST_LOG'] = 'warn'

# Simple problem
config = {
    'n_var': 4,
    'n_obj': 2
}
problem = rust_module.pymoo_interface.create_gnbg_problem(config, "debug")

print("=== ROOT CAUSE ANALYSIS ===")

# The hypothesis: -100 input -> -99 objective means 1 + (-100) = -99
# This suggests distance variables are not being normalized

print("\n1. Testing normalization hypothesis:")
X_test = np.array([[-100.0, -100.0, -100.0, -100.0]], dtype=np.float32)
F_test = problem._evaluate(X_test)['F']
print(f"Input all -100: {X_test[0]}")
print(f"Output: {F_test[0]}")
print(f"Second objective = {F_test[0][1]} (should be close to 1.0 if normalized properly)")

print("\n2. Manual normalization check:")
# Manually normalize -100 -> 0
normalized = [(-100 + 100) / 200.0] * 4  # Should all be 0.0
print(f"Manual normalization of -100: {normalized}")

# Test with those values
X_manual = np.array([normalized], dtype=np.float32)
F_manual = problem._evaluate(X_manual)['F']
print(f"Result with pre-normalized values: {F_manual[0]}")

print("\n3. Distance variable calculation test:")
# If we have 4 variables and 2 objectives:
# n_position = 2 * (2-1) = 2
# n_distance = 4 - 2 = 2
# So distance vars would be [var2, var3] = [0.0, 0.0] after normalization
# Combined distance should be 1.0 + average([0.0, 0.0]) = 1.0
print("Expected: 2 position vars, 2 distance vars")
print("Distance vars after normalization should be [0.0, 0.0]") 
print("Combined distance should be 1.0 + 0.0 = 1.0")
print("But we're getting -99, which suggests raw value -100 is being used")

print("\n4. Checking if issue is with splitter:")
# Test with values that would clearly show if raw vs normalized
X_mixed = np.array([[-100.0, -50.0, 0.0, 100.0]], dtype=np.float32)
F_mixed = problem._evaluate(X_mixed)['F']
print(f"Mixed input: {X_mixed[0]}")
print(f"Output: {F_mixed[0]}")
# If using raw values, distance average would be (0 + 100)/2 = 50, so second obj ~ 51
# If using normalized, distance average would be (0.5 + 1.0)/2 = 0.75, so second obj ~ 1.75

if F_mixed[0][1] > 10:
    print("CONCLUSION: Raw values being used in distance calculation!")
else:
    print("CONCLUSION: Normalized values being used correctly")

print("\n5. GPU vs CPU test:")
# Test if disabling GPU makes a difference  
problem.set_gpu_enabled(False)
X_cpu = np.array([[-100.0, -100.0, -100.0, -100.0]], dtype=np.float32)
F_cpu = problem._evaluate(X_cpu)['F']
print(f"CPU mode result: {F_cpu[0]}")
print(f"Same as GPU result: {np.allclose(F_cpu[0], F_test[0])}")