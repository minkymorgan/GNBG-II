#!/usr/bin/env python3
"""
Debug the evaluation pipeline step by step
"""

import numpy as np
import gnbg_gpu.gnbg_gpu as rust_module

print("🔧 Pipeline Debugging")
print("=" * 30)

# Create a simple problem for debugging
pymoo_interface = rust_module.pymoo_interface

# Test with different numbers of position vs distance variables
configs = [
    {'n_var': 2, 'n_obj': 2},  # All position variables (no distance transformation)
    {'n_var': 3, 'n_obj': 2},  # 1 distance variable  
    {'n_var': 5, 'n_obj': 2},  # 3 distance variables
    {'n_var': 5, 'n_obj': 2, 'wfg': {'problem': 1, 'n_obj': 2}},  # WFG1 with transformations
]

problematic_solutions = [
    [-100.0, -100.0, -100.0, -100.0, -100.0],
    [-50.0, -50.0, -50.0, -50.0, -50.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]

for config_idx, config in enumerate(configs):
    print(f"\n📋 Config {config_idx + 1}: {config}")
    
    try:
        problem = pymoo_interface.create_gnbg_problem(config, f"debug_{config_idx}")
        problem.set_gpu_enabled(False)  # Use CPU for easier debugging
        
        stats = problem.get_stats()
        print(f"   Position vars: {stats['n_position_vars']}, Distance vars: {stats['n_distance_vars']}")
        
        for sol_idx, solution in enumerate(problematic_solutions):
            # Only use as many variables as the problem expects
            test_solution = solution[:config['n_var']]
            
            try:
                objectives = problem.evaluate_single(test_solution)
                finite = np.all(np.isfinite(objectives))
                print(f"   Sol {sol_idx}: {test_solution} → {objectives} (finite: {finite})")
            except Exception as e:
                print(f"   Sol {sol_idx}: {test_solution} → ERROR: {e}")
                
    except Exception as e:
        print(f"   Config creation failed: {e}")

print(f"\n🎯 Analysis Complete")