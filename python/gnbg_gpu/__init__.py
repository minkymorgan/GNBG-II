"""
GNBG-GPU: GPU-accelerated GNBG-II benchmark suite
Copyright (C) 2025 Andrew Morgan <minkymorgan@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .wrapper import GNBG, create_gnbg_suite

try:
    from .gnbg_gpu import GNBGGpu  # Single-objective GPU evaluator
    # Import multi-objective components from the compiled Rust module
    from .gnbg_gpu import pymoo_interface
    GNBGMultiObjectiveProblem = pymoo_interface.GNBGMultiObjectiveProblem
    create_gnbg_problem = pymoo_interface.create_gnbg_problem
    check_algorithm_compatibility = pymoo_interface.check_algorithm_compatibility
    estimate_performance = pymoo_interface.estimate_performance
    _rust_module_available = True
except ImportError:
    # If the compiled module isn't available, create placeholders
    GNBGGpu = None
    GNBGMultiObjectiveProblem = None
    create_gnbg_problem = None
    check_algorithm_compatibility = None
    estimate_performance = None
    _rust_module_available = False

__all__ = [
    'GNBG', 'GNBGGpu', 'create_gnbg_suite',
    'GNBGMultiObjectiveProblem', 'create_gnbg_problem', 
    'check_algorithm_compatibility', 'estimate_performance'
]
__version__ = '0.1.0'