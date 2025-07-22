"""
Python wrapper for GNBG-GPU that provides a compatible interface with the original Python implementation
"""

import numpy as np
from typing import Union, Optional, Tuple
try:
    # Import from the compiled Rust module (gnbg_gpu.cpython-311-darwin.so in same directory)
    from .gnbg_gpu import GNBGGpu
except ImportError:
    try:
        # Alternative: try direct import if installed globally
        import gnbg_gpu as _rust_module
        GNBGGpu = _rust_module.GNBGGpu
    except ImportError:
        raise ImportError("GNBG-GPU Rust module not found. Please build with 'maturin develop --features python'")


class GNBG:
    """
    GPU-accelerated GNBG benchmark class with interface compatible with the original Python implementation.
    
    Parameters
    ----------
    problem_index : int
        Problem index from 1 to 24
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: True). Falls back to CPU if GPU is not available.
    """
    
    def __init__(self, problem_index: int, use_gpu: bool = True):
        if problem_index < 1 or problem_index > 24:
            raise ValueError(f"Problem index must be between 1 and 24, got {problem_index}")
        
        self._gpu_backend = GNBGGpu(problem_index, use_gpu)
        
        # Mirror the original Python implementation's attributes
        self.MaxEvals = self._gpu_backend.max_evals
        self.AcceptanceThreshold = self._gpu_backend.acceptance_threshold
        self.Dimension = self._gpu_backend.dimension
        self.MinCoordinate = self._gpu_backend.min_coordinate
        self.MaxCoordinate = self._gpu_backend.max_coordinate
        self.OptimumValue = self._gpu_backend.optimum_value
        self.OptimumPosition = self._gpu_backend.optimum_position
        
        # Dynamic attributes that update
        self._update_dynamic_attrs()
    
    def _update_dynamic_attrs(self):
        """Update dynamic attributes from the backend"""
        self.FuncEvals = self._gpu_backend.fe_count
        self.BestFoundResult = self._gpu_backend.best_found_result
        self.AcceptanceReachPoint = self._gpu_backend.acceptance_reach_point
        self.FEhistory = self._gpu_backend.fe_history
    
    def fitness(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Evaluate fitness for one or more solutions.
        
        Parameters
        ----------
        X : array-like of shape (n_solutions, dimension) or (dimension,)
            Solution(s) to evaluate. If 1D array, it's treated as a single solution.
        
        Returns
        -------
        result : np.ndarray
            Fitness values. Shape (n_solutions,) for multiple solutions or scalar for single solution.
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Handle single solution case
        if len(X.shape) < 2:
            X = X.reshape(1, -1)
            result = self._gpu_backend.fitness(X)
            self._update_dynamic_attrs()
            return result[0]
        
        result = self._gpu_backend.fitness(X)
        self._update_dynamic_attrs()
        return result
    
    def reset(self):
        """Reset evaluation counters and history"""
        self._gpu_backend.reset()
        self._update_dynamic_attrs()
    
    @property
    def using_gpu(self) -> bool:
        """Check if GPU acceleration is being used"""
        return self._gpu_backend.using_gpu
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the search space bounds.
        
        Returns
        -------
        lower_bounds : np.ndarray
            Lower bounds for each dimension
        upper_bounds : np.ndarray
            Upper bounds for each dimension
        """
        lower = np.full(self.Dimension, self.MinCoordinate)
        upper = np.full(self.Dimension, self.MaxCoordinate)
        return lower, upper
    
    def __repr__(self) -> str:
        return (f"GNBG(dimension={self.Dimension}, max_evals={self.MaxEvals}, "
                f"using_gpu={self.using_gpu})")


def create_gnbg_suite(use_gpu: bool = True) -> dict:
    """
    Create all 24 GNBG problems.
    
    Parameters
    ----------
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: True)
    
    Returns
    -------
    problems : dict
        Dictionary mapping problem names (f1-f24) to GNBG instances
    """
    problems = {}
    for i in range(1, 25):
        try:
            problems[f'f{i}'] = GNBG(i, use_gpu=use_gpu)
        except Exception as e:
            print(f"Failed to load problem f{i}: {e}")
    return problems