"""
Multi-Objective Optimization Interface for GNBG-II

This module provides high-level Python interfaces for creating and using
GNBG multi-objective problems with PyMOO algorithms, following the patterns
observed in pyZenkai benchmark templates.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any

try:
    from .gnbg_gpu import pymoo_interface
    _GNBGMultiObjectiveProblem = pymoo_interface.GNBGMultiObjectiveProblem
    _create_gnbg_problem = pymoo_interface.create_gnbg_problem
    _check_algorithm_compatibility = pymoo_interface.check_algorithm_compatibility
    _estimate_performance = pymoo_interface.estimate_performance
    _rust_available = True
except ImportError:
    _rust_available = False
    _GNBGMultiObjectiveProblem = None
    _create_gnbg_problem = None
    _check_algorithm_compatibility = None
    _estimate_performance = None


class GNBGMultiObjectiveProblem:
    """
    High-level Python wrapper for GNBG Multi-Objective Problems
    
    This class provides a convenient interface for creating multi-objective
    optimization problems that work seamlessly with PyMOO algorithms.
    
    Examples:
        >>> # Create a 3-objective WFG1-style problem
        >>> problem = GNBGMultiObjectiveProblem.wfg1(n_var=10, n_obj=3)
        >>> 
        >>> # Create a custom problem
        >>> problem = GNBGMultiObjectiveProblem({
        ...     'n_var': 20,
        ...     'n_obj': 5,
        ...     'wfg': {'problem': 9, 'n_obj': 5}
        ... })
        >>> 
        >>> # Use with PyMOO algorithms
        >>> from pymoo.algorithms.moo.nsga2 import NSGA2
        >>> from pymoo.optimize import minimize
        >>> algorithm = NSGA2(pop_size=100)
        >>> result = minimize(problem, algorithm, ('n_gen', 100), seed=1)
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize a GNBG multi-objective problem
        
        Args:
            config: Problem configuration dictionary
                   - n_var: Number of variables
                   - n_obj: Number of objectives (optional, defaults to n_var)
                   - wfg: WFG configuration {'problem': X, 'n_obj': Y}
                   - gnbg2: GNBG2 functions list (future extension)
            name: Optional problem name for identification
            
        Raises:
            ImportError: If Rust module is not available
            ValueError: If configuration is invalid
        """
        if not _rust_available:
            raise ImportError(
                "GNBG multi-objective Rust module is not available. "
                "Please compile with 'maturin develop --features python'"
            )
        
        self._config = config.copy()
        self._name = name
        self._problem = _create_gnbg_problem(config, name)
    
    @classmethod
    def wfg1(cls, n_var: int, n_obj: int, name: Optional[str] = None) -> 'GNBGMultiObjectiveProblem':
        """
        Create a WFG1-style problem with polynomial and multimodal transformations
        
        Args:
            n_var: Number of decision variables
            n_obj: Number of objectives
            name: Optional problem name
            
        Returns:
            GNBGMultiObjectiveProblem configured as WFG1
        """
        config = {
            'n_var': n_var,
            'n_obj': n_obj,
            'wfg': {'problem': 1, 'n_obj': n_obj}
        }
        return cls(config, name or f"wfg1_{n_obj}obj_{n_var}var")
    
    @classmethod  
    def wfg2(cls, n_var: int, n_obj: int, name: Optional[str] = None) -> 'GNBGMultiObjectiveProblem':
        """
        Create a WFG2-style problem with polynomial and non-separable transformations
        
        Args:
            n_var: Number of decision variables
            n_obj: Number of objectives  
            name: Optional problem name
            
        Returns:
            GNBGMultiObjectiveProblem configured as WFG2
        """
        config = {
            'n_var': n_var,
            'n_obj': n_obj,
            'wfg': {'problem': 2, 'n_obj': n_obj}
        }
        return cls(config, name or f"wfg2_{n_obj}obj_{n_var}var")
    
    @classmethod
    def wfg3(cls, n_var: int, n_obj: int, name: Optional[str] = None) -> 'GNBGMultiObjectiveProblem':
        """
        Create a WFG3-style problem with polynomial, non-separable, and linear shapes
        
        Args:
            n_var: Number of decision variables
            n_obj: Number of objectives
            name: Optional problem name
            
        Returns:
            GNBGMultiObjectiveProblem configured as WFG3
        """
        config = {
            'n_var': n_var,
            'n_obj': n_obj,
            'wfg': {'problem': 3, 'n_obj': n_obj}
        }
        return cls(config, name or f"wfg3_{n_obj}obj_{n_var}var")
    
    @classmethod
    def custom(cls, n_var: int, n_obj: int, name: Optional[str] = None) -> 'GNBGMultiObjectiveProblem':
        """
        Create a custom multi-objective problem with default settings
        
        Args:
            n_var: Number of decision variables
            n_obj: Number of objectives
            name: Optional problem name
            
        Returns:
            GNBGMultiObjectiveProblem with default configuration
        """
        config = {
            'n_var': n_var,
            'n_obj': n_obj
        }
        return cls(config, name or f"custom_{n_obj}obj_{n_var}var")
    
    @property
    def n_var(self) -> int:
        """Number of decision variables"""
        return self._problem.n_var
    
    @property  
    def n_obj(self) -> int:
        """Number of objectives"""
        return self._problem.n_obj
    
    @property
    def xl(self) -> np.ndarray:
        """Lower bounds for variables (always -100.0 for GNBG problems)"""
        return self._problem.xl
    
    @property
    def xu(self) -> np.ndarray:
        """Upper bounds for variables (always 100.0 for GNBG problems)"""
        return self._problem.xu
    
    @property
    def name(self) -> str:
        """Problem name"""
        return self._problem.name
    
    def _evaluate(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate solutions (PyMOO interface)
        
        Args:
            X: Array of shape (n_solutions, n_var) with solutions to evaluate
            
        Returns:
            Dictionary with 'F' key containing objectives array of shape (n_solutions, n_obj)
        """
        return self._problem._evaluate(X)
    
    def evaluate_single(self, solution: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Evaluate a single solution
        
        Args:
            solution: Decision variables for one solution
            
        Returns:
            Array of objective values
        """
        if isinstance(solution, list):
            solution = np.array(solution, dtype=np.float32)
        return self._problem.evaluate_single(solution)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get problem statistics for monitoring and debugging
        
        Returns:
            Dictionary with problem statistics
        """
        return self._problem.get_stats()
    
    def set_gpu_enabled(self, enabled: bool) -> None:
        """
        Enable or disable GPU acceleration
        
        Args:
            enabled: Whether to use GPU acceleration
        """
        self._problem.set_gpu_enabled(enabled)
    
    def is_gpu_enabled(self) -> bool:
        """
        Check if GPU acceleration is enabled
        
        Returns:
            True if GPU acceleration is enabled
        """
        return self._problem.is_gpu_enabled()
    
    def __repr__(self) -> str:
        return f"GNBGMultiObjectiveProblem(name='{self.name}', n_var={self.n_var}, n_obj={self.n_obj})"


def check_algorithm_compatibility(algorithm: str, n_objectives: int) -> bool:
    """
    Check if an algorithm is compatible with the given number of objectives
    
    Args:
        algorithm: Algorithm name ('NSGA2', 'NSGA3', 'HF1', 'HF3', 'rHF3')
        n_objectives: Number of objectives
        
    Returns:
        True if the algorithm is compatible
        
    Raises:
        ImportError: If Rust module is not available
    """
    if not _rust_available:
        raise ImportError("GNBG multi-objective Rust module is not available")
    
    return _check_algorithm_compatibility(algorithm, n_objectives)


def estimate_performance(n_var: int, n_obj: int, batch_size: int) -> Dict[str, float]:
    """
    Estimate performance for a given problem configuration
    
    Args:
        n_var: Number of variables
        n_obj: Number of objectives
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with performance estimates:
        - estimated_throughput_per_sec: Expected solutions per second
        - estimated_batch_time_ms: Expected time for batch in milliseconds
        - recommended_batch_size: Recommended batch size for this problem
        
    Raises:
        ImportError: If Rust module is not available
    """
    if not _rust_available:
        raise ImportError("GNBG multi-objective Rust module is not available")
    
    return _estimate_performance(n_var, n_obj, batch_size)


def create_problem_suite(
    algorithms: List[str] = None,
    n_objectives: List[int] = None,
    n_variables: List[int] = None,
    wfg_problems: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Create a comprehensive test suite of multi-objective problems
    
    Args:
        algorithms: List of algorithms to test compatibility for
        n_objectives: List of objective counts to test
        n_variables: List of variable counts to test  
        wfg_problems: List of WFG problem numbers to include
        
    Returns:
        List of problem configurations suitable for benchmarking
    """
    if algorithms is None:
        algorithms = ['NSGA2', 'NSGA3', 'HF1', 'HF3', 'rHF3']
    if n_objectives is None:
        n_objectives = [2, 3, 5, 10, 20]
    if n_variables is None:
        n_variables = [10, 20, 30, 50]
    if wfg_problems is None:
        wfg_problems = [1, 2, 3]
    
    suite = []
    
    for n_obj in n_objectives:
        for n_var in n_variables:
            if n_var < n_obj:
                continue  # Skip invalid configurations
                
            # Add WFG problems
            for wfg_num in wfg_problems:
                config = {
                    'name': f'wfg{wfg_num}_{n_obj}obj_{n_var}var',
                    'config': {
                        'n_var': n_var,
                        'n_obj': n_obj,
                        'wfg': {'problem': wfg_num, 'n_obj': n_obj}
                    },
                    'compatible_algorithms': []
                }
                
                # Check algorithm compatibility
                if _rust_available:
                    for alg in algorithms:
                        if check_algorithm_compatibility(alg, n_obj):
                            config['compatible_algorithms'].append(alg)
                else:
                    # Basic compatibility rules when Rust module not available
                    config['compatible_algorithms'] = [
                        alg for alg in algorithms 
                        if not (alg == 'NSGA3' and n_obj < 3)
                    ]
                
                suite.append(config)
            
            # Add custom problem
            config = {
                'name': f'custom_{n_obj}obj_{n_var}var',
                'config': {
                    'n_var': n_var,
                    'n_obj': n_obj
                },
                'compatible_algorithms': []
            }
            
            if _rust_available:
                for alg in algorithms:
                    if check_algorithm_compatibility(alg, n_obj):
                        config['compatible_algorithms'].append(alg)
            else:
                config['compatible_algorithms'] = [
                    alg for alg in algorithms 
                    if not (alg == 'NSGA3' and n_obj < 3)
                ]
            
            suite.append(config)
    
    return suite


# Convenience functions for common use cases
def create_wfg_suite(max_objectives: int = 20) -> List[GNBGMultiObjectiveProblem]:
    """
    Create a suite of WFG problems for testing
    
    Args:
        max_objectives: Maximum number of objectives to test
        
    Returns:
        List of GNBGMultiObjectiveProblem instances
    """
    problems = []
    
    objective_counts = [2, 3, 5, 10]
    if max_objectives > 10:
        objective_counts.extend([15, 20])
    if max_objectives > 20:
        objective_counts.extend([30, 50])
    
    for n_obj in objective_counts:
        if n_obj > max_objectives:
            continue
            
        n_var = max(10, n_obj + 5)  # Ensure n_var >= n_obj
        
        problems.extend([
            GNBGMultiObjectiveProblem.wfg1(n_var, n_obj),
            GNBGMultiObjectiveProblem.wfg2(n_var, n_obj), 
            GNBGMultiObjectiveProblem.wfg3(n_var, n_obj),
        ])
    
    return problems


def benchmark_performance(
    problem: GNBGMultiObjectiveProblem,
    batch_sizes: List[int] = None,
    n_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark performance of a multi-objective problem
    
    Args:
        problem: Problem to benchmark
        batch_sizes: List of batch sizes to test
        n_runs: Number of runs for each batch size
        
    Returns:
        Dictionary with benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [10, 100, 1000, 5000]
    
    results = {
        'problem_name': problem.name,
        'n_var': problem.n_var,
        'n_obj': problem.n_obj,
        'batch_results': [],
        'gpu_enabled': problem.is_gpu_enabled()
    }
    
    for batch_size in batch_sizes:
        # Generate random solutions
        X = np.random.uniform(-100, 100, (batch_size, problem.n_var)).astype(np.float32)
        
        times = []
        for _ in range(n_runs):
            import time
            start = time.time()
            result = problem._evaluate(X)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        
        results['batch_results'].append({
            'batch_size': batch_size,
            'avg_time_sec': avg_time,
            'std_time_sec': np.std(times),
            'throughput_sol_per_sec': throughput,
            'objectives_computed': result['F'].shape[0] * result['F'].shape[1]
        })
    
    return results