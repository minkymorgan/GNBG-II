// src/data_structures.rs
use bytemuck::{Pod, Zeroable};
use anyhow::Result;

/// Direct port of C++ GNBG class data members
#[derive(Debug, Clone)]
pub struct GNBGProblem {
    pub dimension: usize,
    pub comp_num: usize,
    pub min_coordinate: f64,
    pub max_coordinate: f64,
    pub acceptance_threshold: f64,
    pub optimum_value: f64,
    pub max_evals: usize,
    
    // Component parameters
    pub comp_sigma: Vec<f64>,
    pub lambda: Vec<f64>,
    pub comp_min_pos: Vec<Vec<f64>>,
    pub comp_h: Vec<Vec<f64>>,
    pub mu: Vec<[f64; 2]>,
    pub omega: Vec<[f64; 4]>,
    pub rotation_matrices: Vec<Vec<Vec<f64>>>,
    
    // Global optimum
    pub optimum_position: Vec<f64>,
}

/// GPU-friendly version with f32 and flattened arrays
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GNBGParams {
    pub dimension: u32,
    pub comp_num: u32,
    pub min_coordinate: f32,
    pub max_coordinate: f32,
    pub _padding: [f32; 0], // Ensure 16-byte alignment if needed
}

/// Component parameters for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ComponentParams {
    pub sigma: f32,
    pub lambda: f32,
    pub min_pos_offset: u32,
    pub h_offset: u32,
    pub rotation_offset: u32,
    pub mu_omega_offset: u32,
    pub _padding: [f32; 2],
}

impl GNBGProblem {
    /// Convert to GPU-friendly format
    pub fn to_gpu_format(&self) -> Result<GNBGGpuData> {
        let mut comp_params = Vec::new();
        let mut flattened_data = Vec::new();
        let mut mu_omega_data = Vec::new();
        
        let mut current_offset = 0u32;
        
        for i in 0..self.comp_num {
            let min_pos_offset = current_offset;
            // Add min positions
            for j in 0..self.dimension {
                flattened_data.push(self.comp_min_pos[i][j] as f32);
            }
            current_offset += self.dimension as u32;
            
            let h_offset = current_offset;
            // Add h values
            for j in 0..self.dimension {
                flattened_data.push(self.comp_h[i][j] as f32);
            }
            current_offset += self.dimension as u32;
            
            let rotation_offset = current_offset;
            // Add rotation matrix (flattened)
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    flattened_data.push(self.rotation_matrices[i][j][k] as f32);
                }
            }
            current_offset += (self.dimension * self.dimension) as u32;
            
            // Store mu and omega
            let mu_omega_offset = mu_omega_data.len() as u32;
            mu_omega_data.push([
                self.mu[i][0] as f32,
                self.mu[i][1] as f32,
                self.omega[i][0] as f32,
                self.omega[i][1] as f32,
            ]);
            mu_omega_data.push([
                self.omega[i][2] as f32,
                self.omega[i][3] as f32,
                0.0,
                0.0,
            ]);
            
            comp_params.push(ComponentParams {
                sigma: self.comp_sigma[i] as f32,
                lambda: self.lambda[i] as f32,
                min_pos_offset,
                h_offset,
                rotation_offset,
                mu_omega_offset,
                _padding: [0.0; 2],
            });
        }
        
        Ok(GNBGGpuData {
            params: GNBGParams {
                dimension: self.dimension as u32,
                comp_num: self.comp_num as u32,
                min_coordinate: self.min_coordinate as f32,
                max_coordinate: self.max_coordinate as f32,
                _padding: [],
            },
            component_params: comp_params,
            flattened_data,
            mu_omega_data,
            optimum_value: self.optimum_value as f32,
        })
    }
}

pub struct GNBGGpuData {
    pub params: GNBGParams,
    pub component_params: Vec<ComponentParams>,
    pub flattened_data: Vec<f32>,
    pub mu_omega_data: Vec<[f32; 4]>,
    pub optimum_value: f32,
}
