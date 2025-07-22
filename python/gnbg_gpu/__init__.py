"""
GNBG-GPU: GPU-accelerated GNBG-II benchmark suite
"""

from .wrapper import GNBG, create_gnbg_suite

try:
    from gnbg_gpu import GNBGGpu  # This will be the compiled Rust module
except ImportError:
    # If the compiled module isn't available, create a placeholder
    GNBGGpu = None

__all__ = ['GNBG', 'GNBGGpu', 'create_gnbg_suite']
__version__ = '0.1.0'