# GNBG-II Multi-Objective API Specification

## Overview

The GNBG-II Multi-Objective extension provides GPU-accelerated multi-objective optimization capabilities with WFG-style position-distance variable paradigms, transformation pipelines, and shape functions. This API enables seamless integration with optimization frameworks like PyMOO while maintaining high performance through GPU compute shaders.

## Core Architecture

### Multi-Objective Problem Structure

```rust
pub struct GNBGMultiObjective {
    pub dimension: u32,
    pub n_objectives: u32,
    pub splitter: PositionDistanceSplitter,
    pub transformation_pipeline: TransformationPipeline,
    pub shape_executor: ShapeFunctionExecutor,
    pub use_gpu: bool,
}
```

## Builder Pattern API

### GNBGMOBuilder

The primary interface for creating multi-objective problems using a fluent builder pattern.

#### Construction

```rust
let builder = GNBGMOBuilder::new()
    .dimension(20)                    // Total variables
    .objectives(5)                    // Number of objectives
    .split_strategy(SplitStrategy::WFGStandard)
    .gpu(true)                       // Enable GPU acceleration
    .cache(true);                    // Enable shape function caching
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `new()` | Create new builder | None | `GNBGMOBuilder` |
| `dimension(u32)` | Set problem dimension | `dimension: u32` | `Self` |
| `objectives(u32)` | Set number of objectives | `n_objectives: u32` | `Self` |
| `split_strategy(SplitStrategy)` | Set variable splitting | `strategy: SplitStrategy` | `Self` |
| `add_transformation(TransformationType, VariableRange)` | Add transform stage | `transform, range` | `Self` |
| `add_shape(ShapeFunction)` | Add shape function | `shape: ShapeFunction` | `Self` |
| `gpu(bool)` | Enable/disable GPU | `enabled: bool` | `Self` |
| `cache(bool)` | Enable/disable caching | `enabled: bool` | `Self` |
| `build()` | Create problem instance | None | `Result<GNBGMultiObjective>` |

#### WFG Presets

```rust
// WFG1-style: Polynomial + MultiModal transforms, Convex shape
let wfg1 = GNBGMOBuilder::wfg1_preset(20, 5).build()?;

// WFG2-style: Polynomial + NonSeparable transforms, Convex shape  
let wfg2 = GNBGMOBuilder::wfg2_preset(20, 5).build()?;

// WFG3-style: Polynomial + NonSeparable transforms, Linear shape
let wfg3 = GNBGMOBuilder::wfg3_preset(20, 5).build()?;
```

## Position-Distance Variable Splitting

### SplitStrategy

Controls how variables are split between position and distance roles.

```rust
pub enum SplitStrategy {
    WFGStandard,                     // k = 2*(M-1), standard WFG approach
    Custom(u32),                     // Custom k value
    Proportional(f32),               // k = ratio * n_variables  
    Adaptive {                       // Adaptive optimization-driven splitting
        min_k: u32,
        max_k: u32,
        optimization_target: OptimizationTarget,
    },
}

pub enum OptimizationTarget {
    MinimizeObjectives,              // Minimize number of objectives
    MaximizeDiversity,               // Maximize solution diversity
    BalanceComplexity,               // Balance problem complexity
}
```

### PositionDistanceSplitter

```rust
impl PositionDistanceSplitter {
    pub fn new(n_variables: u32, n_objectives: u32, strategy: SplitStrategy) -> Result<Self>;
    pub fn n_position(&self) -> u32;
    pub fn n_distance(&self) -> u32;
    pub fn split_variables(&self, solutions: &[f32]) -> Result<(Vec<f32>, Vec<f32>)>;
}
```

## Transformation Pipeline

### TransformationType

Available transformation functions for distance variables.

```rust
pub enum TransformationType {
    Bias { alpha: f32 },             // Bias transformation: y^alpha
    Deceptive { a: f32, b: f32, c: f32 }, // WFG deceptive transformation
    MultiModal { A: f32, B: f32, C: f32 }, // Multi-modal landscape
    Polynomial { alpha: f32 },        // Polynomial transformation
    Shift { shift: f32 },            // Simple shift transformation
    NonSeparable { A: u32 },         // Non-separable reduction
}
```

### VariableRange

Specifies which variables a transformation applies to.

```rust
pub enum VariableRange {
    All,                             // All variables
    Position,                        // Only position variables
    Distance,                        // Only distance variables  
    Range(usize, usize),             // Specific range [start, end)
    Indices(Vec<usize>),             // Specific variable indices
}
```

### TransformationPipeline

```rust
impl TransformationPipeline {
    pub fn new(stages: Vec<TransformationStage>) -> Result<Self>;
    pub fn add_stage(&mut self, stage: TransformationStage) -> Result<()>;
    pub fn stage_count(&self) -> u32;
    pub async fn apply_gpu_batch(&self, solutions: &[f32], n_position: u32) -> Result<Vec<f32>>;
    pub fn apply_cpu(&self, solution: &mut [f32], n_position: usize) -> Result<()>;
}
```

## Shape Functions

### ShapeFunction

Defines the geometry of the Pareto front in objective space.

```rust
pub enum ShapeFunction {
    Linear,                          // Linear Pareto front (hyperplane)
    Convex,                         // Convex front (sphere-like)
    Concave,                        // Concave front (inverted sphere)
    Mixed {                         // Mixed convex/concave regions
        transition_points: Vec<f32>
    },
    Disconnected {                  // Disconnected front with gaps
        gaps: Vec<(f32, f32)>
    },
}
```

### ShapeFunctionExecutor

```rust
impl ShapeFunctionExecutor {
    pub fn new(shape_functions: Vec<ShapeFunction>) -> Self;
    pub fn with_cache(self) -> Self;
    pub fn apply_cpu(&self, position_vars: &[f32], n_position: u32) -> Result<Vec<f32>>;
    pub async fn apply_gpu(&self, position_vars: &[f32], n_position: u32) -> Result<Vec<f32>>;
}
```

## Evaluation API

### Problem Evaluation

```rust
impl GNBGMultiObjective {
    // Evaluate single solution
    pub async fn evaluate_single(&self, solution: &[f32]) -> Result<Vec<f32>>;
    
    // Evaluate batch of solutions (preferred for performance)
    pub async fn evaluate_batch(&self, solutions: &[f32]) -> Result<Vec<f32>>;
}
```

### Input Format

Solutions are provided as flattened arrays:
```rust
// Single solution: [var1, var2, var3, ..., varN]
let solution = vec![0.1, 0.2, 0.3, 0.4, 0.5];

// Batch of solutions: [sol1_var1, sol1_var2, ..., sol1_varN, sol2_var1, ...]
let batch = vec![
    0.1, 0.2, 0.3, 0.4, 0.5,  // Solution 1
    0.6, 0.7, 0.8, 0.9, 1.0,  // Solution 2
];
```

### Output Format

Objectives are returned as flattened arrays:
```rust
// Single solution: [obj1, obj2, ..., objM]  
let objectives = problem.evaluate_single(&solution).await?;

// Batch: [sol1_obj1, sol1_obj2, ..., sol1_objM, sol2_obj1, ...]
let batch_objectives = problem.evaluate_batch(&batch).await?;
```

## GPU Acceleration

### Fused Transformation Pipeline

The GPU implementation uses a single compute shader dispatch for optimal performance:

```rust
pub struct FusedTransformationPipeline {
    pub pipeline: TransformationPipeline,
    pub compute_pipeline: Option<wgpu::ComputePipeline>,
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub transform_buffer: Option<wgpu::Buffer>,
    pub params_buffer: Option<wgpu::Buffer>,
    pub max_variables: u32,
}

impl FusedTransformationPipeline {
    pub fn new(pipeline: TransformationPipeline) -> Self;
    pub async fn initialize_gpu(&mut self, device: &wgpu::Device) -> Result<()>;
    pub async fn apply_gpu(&self, device: &wgpu::Device, queue: &wgpu::Queue, 
                          solutions: &[f32], n_position: u32) -> Result<Vec<f32>>;
    pub fn apply_cpu(&self, solutions: &mut [f32], n_position: usize) -> Result<()>;
}
```

### Performance Characteristics

| Configuration | Target Performance | Memory Usage |
|---------------|-------------------|--------------|
| 5 objectives, 20 variables | 40,000+ solutions/sec | ~100MB GPU |
| 100 objectives, 200 variables | 5,000+ solutions/sec | ~500MB GPU |
| 500 objectives, 1000 variables | 2,000+ solutions/sec | ~2GB GPU |
| 1000+ objectives | Variable | Variable |

## Error Handling

### GNBGMOError

Comprehensive error types for multi-objective operations:

```rust
pub enum GNBGMOError {
    InvalidConfiguration(String),
    GpuExecutionError(String),
    DimensionMismatch { expected: usize, actual: usize },
    UnsupportedTransformation(String),
    MemoryPoolError(String),
    ShapeFunctionError(String),
}
```

## Memory Management

### GpuMemoryPool

For large-scale optimization (>100K solutions):

```rust
pub struct GpuMemoryPool {
    pool_size: u64,
    available_buffers: Vec<wgpu::Buffer>,
    buffer_size: u64,
}

impl GpuMemoryPool {
    pub fn new(pool_size: u64, buffer_size: u64) -> Self;
    pub fn initialize(&mut self, device: &wgpu::Device) -> Result<()>;
    pub fn get_buffer(&mut self, device: &wgpu::Device, size: u64) -> Result<wgpu::Buffer>;
    pub fn return_buffer(&mut self, buffer: wgpu::Buffer);
}
```

## Complete Usage Examples

### Basic Multi-Objective Problem

```rust
use gnbg_gpu::multi_objective::*;

// Create a 3-objective problem with custom transformations
let problem = GNBGMOBuilder::new()
    .dimension(10)
    .objectives(3)
    .add_transformation(
        TransformationType::Polynomial { alpha: 0.02 },
        VariableRange::Distance
    )
    .add_transformation(
        TransformationType::MultiModal { A: 5.0, B: 10.0, C: 1.0 },
        VariableRange::Distance
    )
    .add_shape(ShapeFunction::Convex)
    .add_shape(ShapeFunction::Convex) 
    .add_shape(ShapeFunction::Concave)
    .build()?;

// Evaluate solutions
let solutions = vec![
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,  // Solution 1
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  // Solution 2
];

let objectives = problem.evaluate_batch(&solutions).await?;
println!("Objectives: {:?}", objectives);  // 6 values: [sol1_obj1, sol1_obj2, sol1_obj3, sol2_obj1, sol2_obj2, sol2_obj3]
```

### WFG-Style Problem

```rust
// Create WFG1-style problem (recommended for most use cases)
let wfg1 = GNBGMOBuilder::wfg1_preset(20, 5).build()?;

// Single solution evaluation
let solution = (0..20).map(|i| (i as f32) / 19.0).collect::<Vec<f32>>();
let objectives = wfg1.evaluate_single(&solution).await?;

assert_eq!(objectives.len(), 5);  // 5 objectives
for &obj in &objectives {
    assert!(obj >= 0.0);  // All objectives should be non-negative
}
```

### High-Performance Batch Evaluation

```rust
// Large batch evaluation for optimization algorithms
let problem = GNBGMOBuilder::wfg2_preset(50, 10)
    .gpu(true)
    .cache(true)
    .build()?;

// Generate 1000 random solutions
let batch_size = 1000;
let dimension = 50;
let mut solutions = Vec::with_capacity(batch_size * dimension);

for _ in 0..batch_size {
    for _ in 0..dimension {
        solutions.push(rand::random::<f32>());
    }
}

// Evaluate entire batch on GPU
let start = std::time::Instant::now();
let objectives = problem.evaluate_batch(&solutions).await?;
let duration = start.elapsed();

println!("Evaluated {} solutions in {:?}", batch_size, duration);
println!("Performance: {:.0} solutions/sec", batch_size as f64 / duration.as_secs_f64());

assert_eq!(objectives.len(), batch_size * 10);  // batch_size × 10 objectives
```

### Adaptive Variable Splitting

```rust
// Use adaptive splitting for complex problems
let problem = GNBGMOBuilder::new()
    .dimension(100)
    .objectives(20)
    .split_strategy(SplitStrategy::Adaptive {
        min_k: 20,
        max_k: 80,
        optimization_target: OptimizationTarget::BalanceComplexity,
    })
    .build()?;

println!("Position variables: {}", problem.splitter.n_position());
println!("Distance variables: {}", problem.splitter.n_distance());
```

## Integration with PyMOO

The API is designed for seamless PyMOO integration (Python bindings coming soon):

```python
# Future PyMOO integration (when Python bindings are complete)
import pymoo
from gnbg_gpu import GNBGMOBuilder

# Create GNBG multi-objective problem
problem = GNBGMOBuilder.wfg1_preset(20, 5).build()

# Use with PyMOO algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
algorithm = NSGA2()

# Problem wrapper (to be implemented)
pymoo_problem = GNBGPyMOOProblem(problem)
result = algorithm.minimize(pymoo_problem)
```

## Performance Tuning

### GPU Optimization Settings

```rust
// For maximum throughput
let problem = GNBGMOBuilder::new()
    .dimension(50)
    .objectives(10)
    .gpu(true)              // Enable GPU acceleration
    .cache(true)            // Enable shape function caching
    .split_strategy(SplitStrategy::WFGStandard)  // Efficient standard splitting
    .build()?;

// Batch sizes: Optimal batch sizes depend on GPU memory
// - Small GPU (4GB): 1,000-10,000 solutions
// - Large GPU (16GB+): 50,000-100,000+ solutions
```

### CPU Fallback for Debugging

```rust
// Use CPU mode for debugging and validation
let debug_problem = GNBGMOBuilder::wfg1_preset(10, 3)
    .gpu(false)             // Force CPU evaluation
    .build()?;

// CPU evaluation is synchronous and deterministic
let objectives = debug_problem.evaluate_batch(&solutions).await?;
```

This API specification provides a comprehensive foundation for multi-objective optimization with GNBG-II, supporting both high-performance GPU acceleration and flexible problem configuration through the builder pattern.