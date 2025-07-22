/*
 * GNBG-II GPU-Accelerated Implementation
 * Copyright (C) 2025 Andrew Morgan <minkymorgan@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
pub mod data_structures;
pub mod gpu_executor;
pub mod gpu_context;
pub mod cpu_reference;
pub mod file_loader;
pub mod shaders;
pub mod multi_objective;

#[cfg(feature = "python")]
pub mod python_bindings;

pub use data_structures::*;
pub use gpu_executor::*;
pub use gpu_context::*;
pub use cpu_reference::*;
pub use file_loader::*;

// Re-export multi-objective types for convenience
pub use multi_objective::{GNBGMOBuilder, GNBGMultiObjective, OptimizationTarget};
#[cfg(feature = "python")]
pub use multi_objective::PyMOOGNBGProblem;
