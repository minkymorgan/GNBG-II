/// Multi-objective extensions for GNBG with WFG-style capabilities
/// 
/// This module provides GPU-accelerated multi-objective optimization benchmarks
/// by extending the existing GNBG framework with position-distance variable
/// paradigms, transformation pipelines, and shape functions.

pub mod position_distance;
pub mod transformations;
pub mod shapes;
pub mod memory_pool;
pub mod pipeline;
pub mod builder;
pub mod mo_problem;
pub mod pymoo_interface;

// Re-export main types for convenience
pub use builder::GNBGMOBuilder;
pub use position_distance::OptimizationTarget;
pub use mo_problem::GNBGMultiObjective;
pub use position_distance::{PositionDistanceSplitter, SplitStrategy};
pub use transformations::TransformationType;
pub use shapes::ShapeFunction;
pub use pymoo_interface::PyMOOGNBGProblem;

/// Multi-objective specific error types
#[derive(Debug, thiserror::Error)]
pub enum GNBGMOError {
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("GPU execution failed: {0}")]
    GpuExecutionError(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Unsupported transformation: {0}")]
    UnsupportedTransformation(String),
    
    #[error("Memory pool error: {0}")]
    MemoryPoolError(String),
    
    #[error("Shape function error: {0}")]
    ShapeFunctionError(String),
}

pub type Result<T> = std::result::Result<T, GNBGMOError>;