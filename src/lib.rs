// src/lib.rs
pub mod data_structures;
pub mod gpu_executor;
pub mod cpu_reference;
pub mod file_loader;
pub mod shaders;

#[cfg(feature = "python")]
pub mod python_bindings;

pub use data_structures::*;
pub use gpu_executor::*;
pub use cpu_reference::*;
pub use file_loader::*;
