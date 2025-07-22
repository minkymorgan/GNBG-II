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
    from .gnbg_gpu import GNBGGpu  # This will be the compiled Rust module
except ImportError:
    # If the compiled module isn't available, create a placeholder
    GNBGGpu = None

__all__ = ['GNBG', 'GNBGGpu', 'create_gnbg_suite']
__version__ = '0.1.0'