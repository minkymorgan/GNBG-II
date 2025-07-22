// src/file_loader.rs
use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::{Result, Context};
use crate::GNBGProblem;

pub fn load_gnbg_problem(func_num: usize) -> Result<GNBGProblem> {
    let filename = format!("f{}.txt", func_num);
    
    // Try to find the file in several locations
    let possible_paths = vec![
        filename.clone(),
        format!("Python_Implementation/GNBG_Instances.Python-main/{}", filename),
        format!("MATLAB_Implementation/GNBG II- Instance.MATLAB/{}", filename),
        format!("C_Implementation/GNBG-Instance-C-main/{}", filename),
    ];
    
    let mut file = None;
    for path in &possible_paths {
        if let Ok(f) = File::open(path) {
            file = Some(f);
            break;
        }
    }
    
    let file = file.ok_or_else(|| {
        anyhow::anyhow!(
            "Failed to find {} in any of the expected locations: {:?}",
            filename, possible_paths
        )
    })?;
    
    let mut reader = BufReader::new(file);
    let mut lines = Vec::new();
    
    // Read all lines first
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        lines.push(line.trim().to_string());
        line.clear();
    }
    
    let mut line_idx = 0;
    
    // Helper to get next line
    let next_line = |idx: &mut usize| -> Result<String> {
        if *idx >= lines.len() {
            return Err(anyhow::anyhow!("Unexpected end of file"));
        }
        let result = lines[*idx].clone();
        *idx += 1;
        Ok(result)
    };
    
    // Helper to parse a single value
    let read_value = |idx: &mut usize| -> Result<f64> {
        let line_content = next_line(idx)?;
        line_content.parse::<f64>()
            .with_context(|| format!("Failed to parse single value: {}", line_content))
    };
    
    // Helper to parse multiple values from a line
    let read_values = |idx: &mut usize| -> Result<Vec<f64>> {
        let line_content = next_line(idx)?;
        line_content.split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
            .with_context(|| format!("Failed to parse values: {}", line_content))
    };
    
    // Read basic parameters
    let max_evals = read_value(&mut line_idx)? as usize;
    let acceptance_threshold = read_value(&mut line_idx)?;
    let dimension = read_value(&mut line_idx)? as usize;
    let comp_num = read_value(&mut line_idx)? as usize;
    let min_coordinate = read_value(&mut line_idx)?;
    let max_coordinate = read_value(&mut line_idx)?;
    
    // Read component positions (one line per component)
    let mut comp_min_pos = vec![vec![0.0; dimension]; comp_num];
    for i in 0..comp_num {
        let values = read_values(&mut line_idx)?;
        if values.len() != dimension {
            return Err(anyhow::anyhow!(
                "Component position dimension mismatch: expected {}, got {}", 
                dimension, values.len()
            ));
        }
        comp_min_pos[i] = values;
    }
    
    // Read sigma values (one line per component)
    let mut comp_sigma = vec![0.0; comp_num];
    for i in 0..comp_num {
        comp_sigma[i] = read_value(&mut line_idx)?;
    }
    
    // Read H values (one line per component)
    let mut comp_h = vec![vec![0.0; dimension]; comp_num];
    for i in 0..comp_num {
        let values = read_values(&mut line_idx)?;
        if values.len() != dimension {
            return Err(anyhow::anyhow!(
                "Component H dimension mismatch: expected {}, got {}", 
                dimension, values.len()
            ));
        }
        comp_h[i] = values;
    }
    
    // Read Mu values (one line with 2 values)
    let mu_values = read_values(&mut line_idx)?;
    if mu_values.len() != 2 {
        return Err(anyhow::anyhow!(
            "Mu values count mismatch: expected 2, got {}", 
            mu_values.len()
        ));
    }
    let mut mu = vec![[0.0; 2]; comp_num];
    for i in 0..comp_num {
        mu[i][0] = mu_values[0];
        mu[i][1] = mu_values[1];
    }
    
    // Read Omega values - flexible approach to handle different formats
    let mut omega = vec![[0.0; 4]; comp_num];
    for i in 0..comp_num {
        let omega_line = read_values(&mut line_idx)?;
        if omega_line.len() == 4 {
            // All 4 values on one line (like f1)
            for j in 0..4 {
                omega[i][j] = omega_line[j];
            }
        } else if omega_line.len() == 2 {
            // Only 2 values per component, pad with zeros
            omega[i][0] = omega_line[0];
            omega[i][1] = omega_line[1];
            omega[i][2] = 0.0;  // Default
            omega[i][3] = 0.0;  // Default
        } else {
            return Err(anyhow::anyhow!(
                "Omega values format not supported for component {}: got {} values", 
                i, omega_line.len()
            ));
        }
    }
    
    // Read Lambda values (one line per component) 
    let mut lambda = vec![0.0; comp_num];
    for i in 0..comp_num {
        lambda[i] = read_value(&mut line_idx)?;
    }
    
    // Read rotation matrices (dimension lines per component)
    let mut rotation_matrices = vec![vec![vec![0.0; dimension]; dimension]; comp_num];
    for i in 0..comp_num {
        for j in 0..dimension {
            let values = read_values(&mut line_idx)?;
            if values.len() != dimension {
                return Err(anyhow::anyhow!(
                    "Rotation matrix row dimension mismatch: expected {}, got {}", 
                    dimension, values.len()
                ));
            }
            rotation_matrices[i][j] = values;
        }
    }
    
    // Read optimum value
    let optimum_value = read_value(&mut line_idx)?;
    
    // Read optimum position (single line with all dimension values)
    let optimum_position = read_values(&mut line_idx)?;
    if optimum_position.len() != dimension {
        return Err(anyhow::anyhow!(
            "Optimum position dimension mismatch: expected {}, got {}", 
            dimension, optimum_position.len()
        ));
    }
    
    Ok(GNBGProblem {
        dimension,
        comp_num,
        min_coordinate,
        max_coordinate,
        acceptance_threshold,
        optimum_value,
        max_evals,
        comp_sigma,
        lambda,
        comp_min_pos,
        comp_h,
        mu,
        omega,
        rotation_matrices,
        optimum_position,
    })
}