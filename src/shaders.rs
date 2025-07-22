// src/shaders.rs
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

fn asymmetric_transform(val: f32, mu: vec2<f32>, omega: vec4<f32>) -> f32 {
    if (val > 0.0) {
        let log_val = log(val);
        return exp(log_val + mu.x * (sin(omega.x * log_val) + sin(omega.y * log_val)));
    } else if (val < 0.0) {
        let log_val = log(-val);
        return -exp(log_val + mu.y * (sin(omega.z * log_val) + sin(omega.w * log_val)));
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let solution_idx = global_id.x;
    let n_solutions = arrayLength(&solutions) / params.dimension;
    
    if (solution_idx >= n_solutions) {
        return;
    }
    
    let dim = params.dimension;
    var min_fitness = 1e10;
    
    // Evaluate each component
    for (var comp_idx = 0u; comp_idx < params.comp_num; comp_idx++) {
        let comp = components[comp_idx];
        
        // Step 1: Translate (x - min_pos)
        var translated: array<f32, 64>;
        for (var i = 0u; i < dim; i++) {
            let sol_val = solutions[solution_idx * dim + i];
            let min_pos = component_data[comp.min_pos_offset + i];
            translated[i] = sol_val - min_pos;
        }
        
        // Step 2: Apply rotation
        var rotated: array<f32, 64>;
        for (var i = 0u; i < dim; i++) {
            rotated[i] = 0.0;
            for (var j = 0u; j < dim; j++) {
                let rot_val = component_data[comp.rotation_offset + i * dim + j];
                rotated[i] += rot_val * translated[j];
            }
        }
        
        // Step 3: Apply asymmetric transformation
        let mu_omega_idx = comp.mu_omega_offset;
        let mu = vec2<f32>(mu_omega[mu_omega_idx].x, mu_omega[mu_omega_idx].y);
        let omega = vec4<f32>(mu_omega[mu_omega_idx].z, mu_omega[mu_omega_idx].w,
                              mu_omega[mu_omega_idx + 1u].x, mu_omega[mu_omega_idx + 1u].y);
        
        for (var i = 0u; i < dim; i++) {
            rotated[i] = asymmetric_transform(rotated[i], mu, omega);
        }
        
        // Step 4: Compute weighted sum
        var sum = 0.0;
        for (var i = 0u; i < dim; i++) {
            let h = component_data[comp.h_offset + i];
            sum += rotated[i] * rotated[i] * h;
        }
        
        // Step 5: Apply final transformation
        let component_fitness = comp.sigma + pow(sum, comp.lambda);
        
        // Take minimum
        min_fitness = min(min_fitness, component_fitness);
    }
    
    fitness_out[solution_idx] = min_fitness;
}
"#;
