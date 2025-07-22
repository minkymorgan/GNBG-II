#!/usr/bin/env python3
"""
Test normalization behavior specifically
"""

import numpy as np
import gnbg_gpu.gnbg_gpu as rust_module

print("🔍 Normalization Test")
print("=" * 30)

pymoo_interface = rust_module.pymoo_interface

# Test different problem configurations to isolate normalization
configs = [
    # Simple: 2 vars, 2 objectives (n_position=1, n_distance=1)
    {'n_var': 2, 'n_obj': 2, 'name': '2var_2obj'},
    
    # More complex: 5 vars, 2 objectives (n_position=2, n_distance=3) 
    {'n_var': 5, 'n_obj': 2, 'name': '5var_2obj'},
    
    # But without WFG transformations (should be simpler)
]

# Test multiple input ranges to verify normalization
test_ranges = [
    [-100.0, -100.0, -100.0, -100.0, -100.0],   # Lower bound
    [0.0, 0.0, 0.0, 0.0, 0.0],                  # Middle
    [100.0, 100.0, 100.0, 100.0, 100.0],       # Upper bound
]

for config in configs:
    print(f"\n📋 Config: {config['name']} ({config['n_var']} vars, {config['n_obj']} obj)")
    
    try:
        problem = pymoo_interface.create_gnbg_problem(config, config['name'])
        problem.set_gpu_enabled(False)  # CPU only
        
        stats = problem.get_stats()
        print(f"   Position: {stats['n_position_vars']}, Distance: {stats['n_distance_vars']}")
        
        for test_idx, test_range in enumerate(test_ranges):
            solution = test_range[:config['n_var']]  # Use only as many vars as needed
            
            try:
                objectives = problem.evaluate_single(solution)
                finite = np.all(np.isfinite(objectives))
                
                # Expected behavior: all objective values should be finite and in reasonable range
                print(f"   Input range {test_idx}: {solution[0]:.1f} → {objectives} (finite: {finite})")
                
                # Check for suspicious values
                if any(abs(obj) > 1000 for obj in objectives):
                    print(f"      ⚠️  Very large objective values detected!")
                if any(obj < -50 for obj in objectives):
                    print(f"      ⚠️  Very negative objective values detected!")
                    
            except Exception as e:
                print(f"   Input range {test_idx}: {solution[0]:.1f} → ERROR: {e}")
                
    except Exception as e:
        print(f"   Config creation failed: {e}")

print(f"\n🎯 Normalization Test Complete")