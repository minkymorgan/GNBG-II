# GNBG-II Multi-Objective Functions: Technical Specification for GF1-GF24

**Version**: 2025.1  
**Authors**: Andrew Morgan (minkymorgan@gmail.com), based on GNBG-II by Danial Yazdani et al.  
**Target Audience**: Optimization researchers, algorithm developers, competition organizers  
**Classification**: Technical Reference Document

---

## Abstract

This document provides a comprehensive mathematical specification of the 24 GNBG-II functions (F1-F24) adapted for multi-objective optimization as GF1-GF24 (GPU Functions). Each function implements sophisticated mathematical transformations including component-based landscapes, non-linear oscillatory transformations, rotation matrices, and configurable conditioning parameters. The mathematical framework draws inspiration from established benchmarks (CEC, WFG) while introducing novel complexity through piece-wise transformations and multi-component minimization structures.

## 1. Mathematical Foundation

### 1.1 General Function Structure

Each GNBG-II function $GF_i$ where $i \in \{1,2,...,24\}$ follows the mathematical formulation:

$$GF_i(\mathbf{x}) = \min_{k=1}^{K_i} f_k(\mathbf{x})$$

where $K_i$ is the number of components for function $i$, and each component function $f_k$ is defined as:

$$f_k(\mathbf{x}) = \sigma_k + \left(\mathbf{a}_k^T \mathbf{H}_k \mathbf{b}_k\right)^{\lambda_k}$$

### 1.2 Variable Transformation Pipeline

For each component $k$, the evaluation follows a sophisticated transformation sequence:

#### Step 1: Spatial Transformation
$$\mathbf{x}' = \mathbf{x} - \boldsymbol{\mu}_k$$

where $\boldsymbol{\mu}_k \in \mathbb{R}^D$ represents the component-specific minimum position.

#### Step 2: Rotation and Non-linear Transformation
$$\mathbf{a}_k = \mathcal{T}\left((\mathbf{x}')^T \mathbf{R}_k^T, \boldsymbol{\alpha}_k, \boldsymbol{\beta}_k\right)$$
$$\mathbf{b}_k = \mathcal{T}\left(\mathbf{R}_k \mathbf{x}', \boldsymbol{\alpha}_k, \boldsymbol{\beta}_k\right)$$

where:
- $\mathbf{R}_k \in \mathbb{R}^{D \times D}$ is an orthogonal rotation matrix for component $k$
- $\mathcal{T}(\cdot)$ is the non-linear transformation function (detailed in Section 1.3)
- $\boldsymbol{\alpha}_k, \boldsymbol{\beta}_k$ are transformation parameters

#### Step 3: Component Evaluation
$$f_k(\mathbf{x}) = \sigma_k + \left(\mathbf{a}_k^T \text{diag}(\mathbf{h}_k) \mathbf{b}_k\right)^{\lambda_k}$$

where:
- $\sigma_k \in \mathbb{R}$ is the component bias term
- $\mathbf{h}_k \in \mathbb{R}^D$ is the conditioning vector
- $\lambda_k \in \mathbb{R}^+$ is the power scaling exponent

### 1.3 Non-Linear Transformation Function

The transformation function $\mathcal{T}: \mathbb{R}^D \times \mathbb{R}^2 \times \mathbb{R}^4 \rightarrow \mathbb{R}^D$ implements piece-wise oscillatory behavior:

$$\mathcal{T}(\mathbf{x}, \boldsymbol{\alpha}, \boldsymbol{\beta}) = \mathbf{y}$$

where for each component $x_i$:

$$y_i = \begin{cases}
\exp\left(\ln(x_i) + \alpha_1 \left(\sin(\beta_1 \ln(x_i)) + \sin(\beta_2 \ln(x_i))\right)\right) & \text{if } x_i > 0 \\
-\exp\left(\ln(-x_i) + \alpha_2 \left(\sin(\beta_3 \ln(-x_i)) + \sin(\beta_4 \ln(-x_i))\right)\right) & \text{if } x_i < 0 \\
0 & \text{if } x_i = 0
\end{cases}$$

**Mathematical Properties:**
- **Continuity**: $\lim_{x \to 0^+} \mathcal{T}(x) = \lim_{x \to 0^-} \mathcal{T}(x) = 0$
- **Oscillatory behavior**: Sinusoidal terms create multi-modal landscapes
- **Asymmetric scaling**: Parameters $\alpha_1, \alpha_2$ control positive/negative scaling independently
- **Logarithmic compression**: Prevents numerical overflow while preserving landscape structure

## 2. Function Taxonomy and Characteristics

### 2.1 Unimodal Functions (GF1-GF6)

These functions establish baseline optimization challenges with single global optima but varying conditioning and rotation properties.

#### GF1: Well-Conditioned Separable Function
**Mathematical Form:**
$$GF_1(\mathbf{x}) = \sigma_1 + \left(\sum_{i=1}^{30} x_i^2\right)^{\lambda_1}$$

**Parameters:**
- $K_1 = 1$ (single component)
- $\sigma_1 = -1081.984$
- $\mathbf{h}_1 = [1, 1, ..., 1]^T$ (uniform conditioning)
- $\boldsymbol{\alpha}_1 = [0, 0]$, $\boldsymbol{\beta}_1 = [0, 0, 0, 0]$ (no transformation)
- $\lambda_1 = 1.0$ (quadratic form)
- $\mathbf{R}_1 = \mathbf{I}$ (identity matrix, no rotation)

**Characteristics:** Baseline sphere function, separable, unimodal, well-conditioned.

#### GF2: Moderately Conditioned Function  
**Mathematical Form:**
$$GF_2(\mathbf{x}) = \sigma_2 + \left(\mathbf{x}^T \text{diag}(\mathbf{h}_2) \mathbf{x}\right)^{\lambda_2}$$

**Parameters:**
- $K_2 = 1$
- $\sigma_2 = -104.865$ 
- $\mathbf{h}_2 = [10^2, 10^{1.8}, 10^{1.6}, ..., 1]^T$ (moderate conditioning, ratio ≈ 100:1)
- $\boldsymbol{\alpha}_2 = [0, 0]$, $\boldsymbol{\beta}_2 = [0, 0, 0, 0]$
- $\lambda_2 = 1.0$
- $\mathbf{R}_2 = \mathbf{I}$

**Characteristics:** Axis-aligned ellipsoid, moderate ill-conditioning, separable.

#### GF3: Ill-Conditioned Function
**Parameters:**
- $K_3 = 1$
- $\sigma_3 = -51.479$
- $\mathbf{h}_3 = [10^4, 10^{3.7}, 10^{3.4}, ..., 1]^T$ (high conditioning, ratio ≈ 10,000:1)
- $\boldsymbol{\alpha}_3 = [0, 0]$, $\boldsymbol{\beta}_3 = [0, 0, 0, 0]$  
- $\lambda_3 = 1.0$
- $\mathbf{R}_3 = \mathbf{I}$

**Characteristics:** Severely ill-conditioned ellipsoid, numerical challenges, separable.

#### GF4: Non-Separable Rotated Function
**Parameters:**
- $K_4 = 1$
- $\sigma_4 = -930.063$
- $\mathbf{h}_4 = [1, 1, ..., 1]^T$
- $\boldsymbol{\alpha}_4 = [0, 0]$, $\boldsymbol{\beta}_4 = [0, 0, 0, 0]$
- $\lambda_4 = 1.0$
- $\mathbf{R}_4$ = 30×30 orthogonal rotation matrix

**Characteristics:** Rotated sphere, non-separable, tests rotation invariance.

#### GF5: Non-Separable with Complex Conditioning
**Parameters:**
- $K_5 = 1$
- $\sigma_5 = -525.477$
- $\mathbf{h}_5 = [99999.97, 1423.07, 88949.04, ...]^T$ (irregular conditioning)
- $\boldsymbol{\alpha}_5 = [0, 0]$, $\boldsymbol{\beta}_5 = [0, 0, 0, 0]$
- $\lambda_5 = 0.75$ (sub-quadratic scaling)
- $\mathbf{R}_5$ = full rotation matrix

**Characteristics:** Rotated ill-conditioned ellipsoid with irregular eigenvalue distribution.

#### GF6: Rotated and Shifted Complex
**Parameters:**
- $K_6 = 1$
- $\sigma_6 = -337.509$
- $\mathbf{h}_6$ = irregular conditioning vector
- $\boldsymbol{\alpha}_6 = [0, 0]$, $\boldsymbol{\beta}_6 = [0, 0, 0, 0]$
- $\lambda_6 = 1.0$
- $\mathbf{R}_6$ = full rotation matrix
- $\boldsymbol{\mu}_6$ = non-zero shift vector

**Characteristics:** Combines rotation, shifting, and irregular conditioning.

### 2.2 Single-Component Multimodal Functions (GF7-GF15)

These functions introduce oscillatory transformations creating multiple local optima while maintaining single-component structure.

#### GF7: Basic Multimodal Function
**Parameters:**
- $K_7 = 1$
- $\sigma_7 = -874.478$
- $\mathbf{h}_7$ = moderate conditioning
- $\boldsymbol{\alpha}_7 = [0.1, 0.1]$ (weak transformation)
- $\boldsymbol{\beta}_7 = [3.0, 3.0, 3.0, 3.0]$ (low-frequency oscillations)
- $\lambda_7 = 1.0$
- $\mathbf{R}_7$ = rotation matrix

**Characteristics:** Introduces basic multimodality through weak oscillatory transformations.

#### GF9: Highly Multimodal Function
**Parameters:**
- $K_9 = 1$
- $\sigma_9 = -1030.301$
- $\mathbf{h}_9$ = high variation conditioning
- $\boldsymbol{\alpha}_9 = [0.3, 0.3]$ (moderate transformation)
- $\boldsymbol{\beta}_9 = [8.0, 8.0, 8.0, 8.0]$ (medium-frequency oscillations)
- $\lambda_9 = 0.8$
- $\mathbf{R}_9$ = full rotation matrix

**Characteristics:** Increased multimodality, more deceptive landscape.

#### GF12: Complex Multimodal with Strong Transformation
**Parameters:**
- $K_{12} = 1$
- $\sigma_{12} = -401.147$
- $\mathbf{h}_{12} = [68535.28, 19269.26, 83970.64, ...]^T$ (extreme conditioning)
- $\boldsymbol{\alpha}_{12} = [0.5, 0.5]$ (strong transformation)
- $\boldsymbol{\beta}_{12} = [21.40, 37.37, 48.58, 16.03]$ (high-frequency oscillations)
- $\lambda_{12} = 0.5$ (square root scaling)
- $\mathbf{R}_{12}$ = full rotation matrix

**Characteristics:** Combines extreme conditioning with strong oscillatory transformations, highly deceptive.

#### GF15: Maximum Single-Component Complexity
**Parameters:**
- $K_{15} = 1$
- $\sigma_{15} = -624.056$
- Complex parameter combinations achieving maximum single-component difficulty

**Characteristics:** Peak single-component challenge, comprehensive algorithm testing.

### 2.3 Multi-Component Multimodal Functions (GF16-GF24)

These functions implement multiple independent components with min() selection, creating complex multimodal landscapes.

#### GF16: Five-Component Hybrid
**Mathematical Form:**
$$GF_{16}(\mathbf{x}) = \min_{k=1}^{5} f_k(\mathbf{x})$$

**Parameters:**
- $K_{16} = 5$ components
- $\sigma_{16} = [-4208.486, -4208.486, -4208.486, -4208.486, -4208.486]^T$
- Each component has distinct:
  - Rotation matrix $\mathbf{R}_k$  
  - Conditioning vector $\mathbf{h}_k$
  - Transformation parameters $\boldsymbol{\alpha}_k, \boldsymbol{\beta}_k$
  - Minimum position $\boldsymbol{\mu}_k$

**Characteristics:** Five competing basins, complex global structure, basin-hopping challenges.

#### GF20: Dual-Component System
**Mathematical Form:**
$$GF_{20}(\mathbf{x}) = \min_{k=1}^{2} f_k(\mathbf{x})$$

**Parameters:**
- $K_{20} = 2$ components
- $\sigma_{20} = [-1000.000, -1000.000]^T$
- Distinct component parameters creating binary choice optimization

**Characteristics:** Simplest multi-component case, studies basin selection dynamics.

#### GF24: Ultimate Challenge (Seven Components)
**Mathematical Form:**
$$GF_{24}(\mathbf{x}) = \min_{k=1}^{7} f_k(\mathbf{x})$$

**Parameters:**
- $K_{24} = 7$ components (maximum complexity)
- $\sigma_{24} = [-99.917, -99.917, ..., -99.917]^T$ (seven identical baselines)
- Component-specific parameters:
  - $\boldsymbol{\alpha}_k$ values: $[0.44, 0.36], [0.30, 0.23], ...$
  - $\boldsymbol{\beta}_k$ values: $[40.37, 45.97, 30.11, 33.39], ...$
  - $\lambda_k = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]^T$ (strong compression)
  - Seven distinct 30×30 rotation matrices
  - Seven distinct conditioning vectors with extreme variations

**Characteristics:** Maximum GNBG-II complexity, seven competing basins with high-frequency oscillations, ultimate algorithm challenge.

## 3. Multi-Objective Extension Framework

### 3.1 Position-Distance Variable Paradigm

For multi-objective optimization, GNBG functions are extended using the position-distance variable paradigm:

**Variable Decomposition:**
$$\mathbf{x} = [\mathbf{x}_{\text{pos}}, \mathbf{x}_{\text{dist}}]$$

where:
- $\mathbf{x}_{\text{pos}} \in \mathbb{R}^{M-1}$: Position variables (M-1 for M objectives)
- $\mathbf{x}_{\text{dist}} \in \mathbb{R}^{D-(M-1)}$: Distance variables

**Objective Functions:**
$$f_i(\mathbf{x}) = g(\mathbf{x}_{\text{dist}}) \cdot h_i(\mathbf{x}_{\text{pos}})$$

where:
- $g(\mathbf{x}_{\text{dist}}) = 1 + GF_k(\mathbf{x}_{\text{dist}})$ (shape function using GNBG landscape)
- $h_i(\mathbf{x}_{\text{pos}})$ defines the Pareto front geometry

### 3.2 WFG-Inspired Transformations

The multi-objective extension incorporates techniques from the Walking Fish Group (WFG) test suite:

#### Polynomial Bias Transformation
$$t_i = x_i^{\alpha}$$

#### Linear Transformation  
$$t_i = \sum_{j=1}^{n} A_{ij} x_j$$

#### Reduction Transformations
$$t_i = \sum_{j=(i-1)k+1}^{ik} w_j x_j / \sum_{j=(i-1)k+1}^{ik} w_j$$

### 3.3 Pareto Front Geometries

**Type 1: Linear Front**
$$h_i(\mathbf{x}_{\text{pos}}) = \prod_{j=1}^{M-i} x_j \cdot \begin{cases} 1 & \text{if } i = M \\ 1-x_{M-i} & \text{otherwise} \end{cases}$$

**Type 2: Convex Front**  
$$h_i(\mathbf{x}_{\text{pos}}) = \prod_{j=1}^{M-i} \sin(x_j \pi/2) \cdot \begin{cases} 1 & \text{if } i = M \\ \cos(x_{M-i} \pi/2) & \text{otherwise} \end{cases}$$

**Type 3: Concave Front**
$$h_i(\mathbf{x}_{\text{pos}}) = \prod_{j=1}^{M-i} \cos(x_j \pi/2) \cdot \begin{cases} 1 & \text{if } i = M \\ \sin(x_{M-i} \pi/2) & \text{otherwise} \end{cases}$$

## 4. Implementation Details

### 4.1 Numerical Considerations

**Precision Requirements:**
- All calculations performed in double precision (64-bit)
- Transformation function handles x=0 case explicitly
- Logarithmic operations protected against domain errors

**Boundary Handling:**
Midpoint-target constraint handling for violations:
$$x_{\text{corrected}} = \begin{cases}
(x + x_{\text{min}}) / 2 & \text{if } x < x_{\text{min}} \\
(x + x_{\text{max}}) / 2 & \text{if } x > x_{\text{max}} \\
x & \text{otherwise}
\end{cases}$$

### 4.2 GPU Acceleration Architecture

**Batch Processing:**
- Simultaneous evaluation of N solutions: $\mathbf{X} \in \mathbb{R}^{N \times D}$
- Parallel transformation computations
- Vectorized matrix operations

**Memory Layout:**
```
X[batch_size][dimension]           -> Input solutions
RotationMatrices[component][D][D]  -> Component rotation matrices  
ComponentH[component][dimension]   -> Conditioning vectors
TransformParams[component][6]      -> Alpha, Beta parameters
Results[batch_size][n_objectives]  -> Output objectives
```

**Compute Shader Structure:**
```glsl
@group(0) @binding(0) var<storage, read> input_X: array<f32>;
@group(0) @binding(1) var<storage, read> rotation_matrices: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_F: array<f32>;

@compute @workgroup_size(64)
fn evaluate_gnbg(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let solution_idx = global_id.x;
    
    // Load solution
    var x: array<f32, 30>;
    for (var i = 0; i < 30; i++) {
        x[i] = input_X[solution_idx * 30 + i];
    }
    
    // Evaluate each component
    var min_value = 1e10;
    for (var k = 0; k < num_components; k++) {
        let component_value = evaluate_component(x, k);
        min_value = min(min_value, component_value);
    }
    
    // Store result
    output_F[solution_idx] = min_value;
}
```

### 4.3 Performance Characteristics

**Throughput Analysis:**
- **Single-thread CPU**: ~1,000 evaluations/second
- **Multi-thread CPU**: ~10,000 evaluations/second  
- **GPU (WebGPU)**: 600,000+ evaluations/second
- **Speedup factor**: 60-600x over single-thread

**Scaling Behavior:**
- Linear scaling with batch size up to GPU memory limits
- Optimal batch sizes: 1,000-10,000 solutions
- Memory usage: ~4MB per 10,000 solutions (30D, single precision)

## 5. Competition and Benchmarking Guidelines

### 5.1 Standard Experimental Protocol

**Algorithm Testing Requirements:**
- 31 independent runs per function per algorithm
- Evaluation budget: MaxEvals per function (500K-1M)
- Success criterion: |f(x) - f(x*)| < 1e-08
- Termination: Budget exhaustion or success criterion met

**Statistical Analysis:**
- Report mean ± standard deviation of final errors
- Success rate (percentage achieving convergence)
- Convergence curves (error vs. function evaluations)
- Statistical significance testing (Wilcoxon signed-rank)

### 5.2 Multi-Objective Evaluation Metrics

**Hypervolume Indicator:**
$$HV(A) = \mu\left(\bigcup_{\mathbf{a} \in A} [\mathbf{a}, \mathbf{r}]\right)$$

where $\mathbf{r}$ is the reference point and $\mu$ is the Lebesgue measure.

**Inverted Generational Distance (IGD):**
$$IGD(A, P^*) = \frac{1}{|P^*|} \sum_{\mathbf{p} \in P^*} \min_{\mathbf{a} \in A} ||\mathbf{p} - \mathbf{a}||$$

**Additive Epsilon Indicator:**
$$I_{\epsilon+}(A, B) = \inf_{\epsilon \in \mathbb{R}} \forall \mathbf{b} \in B, \exists \mathbf{a} \in A : \mathbf{a} \preceq_{\epsilon} \mathbf{b}$$

### 5.3 Algorithm Requirements

**Constraint Handling:**
- Solutions must satisfy $\mathbf{x} \in [-100, 100]^D$
- Algorithms responsible for boundary constraint satisfaction
- No penalty functions - use direct constraint handling

**Evaluation Counting:**
- Every function call counts toward budget
- Batch evaluations count as individual calls
- Gradient estimations count individual evaluations

## 6. Research Applications and Extensions

### 6.1 Algorithm Development Targets

**Global Optimization Challenges:**
- **Basin identification**: Multi-component functions test global search
- **Local refinement**: Oscillatory transformations require fine-tuned local search
- **Numerical robustness**: Extreme conditioning tests algorithm stability
- **Rotation invariance**: Non-separable functions test coordinate system independence

**Multi-Objective Specific:**
- **Pareto front approximation**: Various front geometries test convergence
- **Diversity maintenance**: Distance variables test solution spread
- **Computational efficiency**: High-dimensional objectives test scalability

### 6.2 Theoretical Analysis Opportunities

**Landscape Analysis:**
- Funnel structure characterization
- Local optima counting and distribution
- Basin of attraction estimation
- Gradient information reliability

**Complexity Theory:**
- Runtime analysis for specific algorithm classes
- Problem hardness quantification
- Search space structure theorems

### 6.3 Practical Extensions

**Industry Applications:**
- Engineering design optimization
- Machine learning hyperparameter tuning  
- Financial portfolio optimization
- Scientific parameter estimation

**Computational Enhancements:**
- Distributed evaluation frameworks
- Adaptive precision computation
- Surrogate model integration
- Interactive optimization interfaces

## 7. Conclusion

The GNBG-II function suite (GF1-GF24) represents a sophisticated benchmark framework combining mathematical rigor with computational efficiency. The systematic progression from unimodal (GF1-GF6) through single-component multimodal (GF7-GF15) to multi-component multimodal (GF16-GF24) provides comprehensive algorithm testing across diverse optimization challenges.

Key contributions include:

1. **Mathematical sophistication**: Component-based min() structures with oscillatory transformations create realistic optimization challenges
2. **Computational efficiency**: GPU acceleration enables large-scale empirical studies  
3. **Multi-objective extension**: Position-distance variable paradigm with WFG-inspired transformations
4. **Systematic difficulty progression**: Organized complexity levels support targeted algorithm development

The framework supports both theoretical analysis and practical algorithm development, making it valuable for optimization research, competition organization, and industrial application development.

Future work should focus on theoretical landscape analysis, adaptive difficulty scaling, and integration with modern machine learning optimization challenges.

---

## References

1. Yazdani, D., et al. "GNBG: A Generalized Numerical Benchmark Generator for Continuous Optimization." arXiv:2312.07034, 2023.
2. Huband, S., et al. "A review of multiobjective test problems and a scalable test problem toolkit." IEEE Transactions on Evolutionary Computation, 2006.
3. Morgan, A. "GPU-Accelerated Multi-Objective Optimization using WebGPU and Rust." Technical Report, 2025.
4. CEC 2025 Competition Guidelines. IEEE Congress on Evolutionary Computation, 2025.

---

**Document Version History:**
- v2025.1: Initial comprehensive specification
- Last updated: July 2025
- Status: Complete technical reference