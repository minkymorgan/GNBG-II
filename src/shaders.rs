// src/shaders.rs

// Optimized GPU compute shader with performance improvements
pub const GNBG_COMPUTE_SHADER: &str = r#"
struct GNBGParams {
    dimension: u32,
    comp_num: u32,
    min_coordinate: f32,
    max_coordinate: f32,
}

struct ComponentParams {
    sigma: f32,
    lambda: f32,
    min_pos_offset: u32,
    h_offset: u32,
    rotation_offset: u32,
    mu_omega_offset: u32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<uniform> params: GNBGParams;
@group(0) @binding(2) var<storage, read> components: array<ComponentParams>;
@group(0) @binding(3) var<storage, read> component_data: array<f32>;
@group(0) @binding(4) var<storage, read> mu_omega: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> fitness_out: array<f32>;

// Optimized asymmetric transformation with reduced branching
fn asymmetric_transform_optimized(val: f32, mu: vec2<f32>, omega: vec4<f32>) -> f32 {
    let abs_val = abs(val);
    if (abs_val < 1e-10) {
        return 0.0;
    }
    
    let log_val = log(abs_val);
    let is_positive = val > 0.0;
    
    // Branchless computation using select()
    let mu_sel = select(mu.y, mu.x, is_positive);
    let omega_sel = select(vec2<f32>(omega.z, omega.w), vec2<f32>(omega.x, omega.y), is_positive);
    let sin_vals = sin(omega_sel * log_val);
    let result_abs = exp(log_val + mu_sel * (sin_vals.x + sin_vals.y));
    
    return select(-result_abs, result_abs, is_positive);
}

// Optimized workgroup size for better GPU occupancy
@compute @workgroup_size(256)  // Increased from 64 for better utilization
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let solution_idx = global_id.x;
    let n_solutions = arrayLength(&solutions) / params.dimension;
    
    if (solution_idx >= n_solutions) {
        return;
    }
    
    let dim = params.dimension;
    let solution_base = solution_idx * dim;
    var min_fitness = 1e10;
    
    // Evaluate each component
    for (var comp_idx = 0u; comp_idx < params.comp_num; comp_idx++) {
        let comp = components[comp_idx];
        
        // Preload mu/omega data for this component
        let mu_omega_idx = comp.mu_omega_offset;
        let mu = vec2<f32>(mu_omega[mu_omega_idx].x, mu_omega[mu_omega_idx].y);
        let omega = vec4<f32>(
            mu_omega[mu_omega_idx].z, 
            mu_omega[mu_omega_idx].w,
            mu_omega[mu_omega_idx + 1u].x, 
            mu_omega[mu_omega_idx + 1u].y
        );
        
        // Combined transformation pipeline for better cache efficiency
        var sum = 0.0;
        
        for (var i = 0u; i < dim; i++) {
            // Step 1 & 2: Translation and rotation combined
            var rotated_val = 0.0;
            for (var j = 0u; j < dim; j++) {
                let translated = solutions[solution_base + j] - component_data[comp.min_pos_offset + j];
                rotated_val += component_data[comp.rotation_offset + i * dim + j] * translated;
            }
            
            // Step 3: Optimized asymmetric transformation
            let transformed = asymmetric_transform_optimized(rotated_val, mu, omega);
            
            // Step 4: Weighted sum accumulation
            let h = component_data[comp.h_offset + i];
            sum += transformed * transformed * h;
        }
        
        // Step 5: Final transformation
        let component_fitness = comp.sigma + pow(sum, comp.lambda);
        
        // Take minimum across components
        min_fitness = min(min_fitness, component_fitness);
    }
    
    fitness_out[solution_idx] = min_fitness;
}
"#;

// Alternative high-performance shader for larger workloads
pub const GNBG_COMPUTE_SHADER_ULTRA: &str = r#"
struct GNBGParams {
    dimension: u32,
    comp_num: u32,
    min_coordinate: f32,
    max_coordinate: f32,
}

struct ComponentParams {
    sigma: f32,
    lambda: f32,
    min_pos_offset: u32,
    h_offset: u32,
    rotation_offset: u32,
    mu_omega_offset: u32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var<storage, read> solutions: array<f32>;
@group(0) @binding(1) var<uniform> params: GNBGParams;
@group(0) @binding(2) var<storage, read> components: array<ComponentParams>;
@group(0) @binding(3) var<storage, read> component_data: array<f32>;
@group(0) @binding(4) var<storage, read> mu_omega: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> fitness_out: array<f32>;

// Ultra-optimized for maximum throughput
fn asymmetric_transform_ultra(val: f32, mu: vec2<f32>, omega: vec4<f32>) -> f32 {
    let abs_val = abs(val);
    if (abs_val < 1e-10) { return 0.0; }
    
    let log_val = log(abs_val);
    let is_pos = val > 0.0;
    
    // Branchless computation
    let mu_sel = select(mu.y, mu.x, is_pos);
    let omega_sel = select(vec2<f32>(omega.z, omega.w), vec2<f32>(omega.x, omega.y), is_pos);
    let sin_vals = sin(omega_sel * log_val);
    let result_abs = exp(log_val + mu_sel * (sin_vals.x + sin_vals.y));
    
    return select(-result_abs, result_abs, is_pos);
}

@compute @workgroup_size(1024)  // Maximum workgroup size for ultra performance
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let solution_idx = global_id.x;
    let n_solutions = arrayLength(&solutions) / params.dimension;
    
    if (solution_idx >= n_solutions) { return; }
    
    let dim = params.dimension;
    let solution_base = solution_idx * dim;
    var min_fitness = 1e10;
    
    for (var comp_idx = 0u; comp_idx < params.comp_num; comp_idx++) {
        let comp = components[comp_idx];
        
        // Ultra-optimized computation with minimal memory access
        var sum = 0.0;
        let mu = vec2<f32>(mu_omega[comp.mu_omega_offset].xy);
        let omega = vec4<f32>(
            mu_omega[comp.mu_omega_offset].z, 
            mu_omega[comp.mu_omega_offset].w,
            mu_omega[comp.mu_omega_offset + 1u].x, 
            mu_omega[comp.mu_omega_offset + 1u].y
        );
        
        // Unrolled loop for maximum performance on small dimensions
        for (var i = 0u; i < dim; i++) {
            // Combined translation and rotation in minimal operations
            var rotated_val = 0.0;
            let solution_val = solutions[solution_base + i];
            
            for (var j = 0u; j < dim; j++) {
                let translated = solutions[solution_base + j] - component_data[comp.min_pos_offset + j];
                rotated_val += component_data[comp.rotation_offset + i * dim + j] * translated;
            }
            
            let transformed = asymmetric_transform_ultra(rotated_val, mu, omega);
            let h = component_data[comp.h_offset + i];
            sum += transformed * transformed * h;
        }
        
        min_fitness = min(min_fitness, comp.sigma + pow(sum, comp.lambda));
    }
    
    fitness_out[solution_idx] = min_fitness;
}
"#;
