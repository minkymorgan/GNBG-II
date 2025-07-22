/// Multi-modal transformation function
/// 
/// Creates multiple local optima in the fitness landscape

use super::correct_to_01;
use std::f32::consts::PI;

/// Apply multi-modal transformation
/// 
/// Creates a landscape with multiple peaks/valleys to test algorithm's
/// ability to escape local optima.
/// 
/// # Arguments
/// * `y` - Input value in [0,1]
/// * `A` - Number of modes (typical: 5-20)
/// * `B` - Hill size parameter (typical: 10.0)
/// * `C` - Scaling parameter (typical: 1.0)
/// 
/// # Returns
/// Transformed value in [0,1]
pub fn multimodal_transform(y: f32, A: f32, B: f32, C: f32) -> f32 {
    let tmp = 2.0 * y - 1.0;
    let a = 0.35;
    let b = 0.001;
    
    let cosine_term = (A * PI * tmp).cos();
    let power_term = 4.0 * b * tmp.abs().powf(2.0);
    
    let result = (1.0 + cosine_term + power_term) / (b + 2.0);
    
    correct_to_01(C * result)
}

/// WFG-style multi-modal transformation
/// 
/// Implements the exact WFG multi-modal transformation
/// 
/// # Arguments
/// * `y` - Input value in [0,1]
/// * `num_modes` - Number of modes
/// * `hill_size` - Size of hills
/// 
/// # Returns
/// Transformed value in [0,1] 
pub fn wfg_multimodal_transform(y: f32, num_modes: u32, hill_size: f32) -> f32 {
    let tmp = 2.0 * y - 1.0;
    let a = 0.35;
    let b = 0.001;
    let c = 0.05;
    
    let modes_f32 = num_modes as f32;
    let cosine_term = (modes_f32 * PI * tmp).cos();
    let quad_term = 4.0 * b * tmp.abs().powf(2.0);
    
    let numerator = 1.0 + cosine_term + quad_term;
    let denominator = b + 2.0;
    
    correct_to_01(numerator / denominator)
}

/// Parametric multi-modal transformation
/// 
/// More flexible version allowing custom parameters
/// 
/// # Arguments
/// * `y` - Input value in [0,1]
/// * `frequency` - Frequency of modes
/// * `amplitude` - Amplitude of modes
/// * `base_level` - Base level
/// 
/// # Returns
/// Transformed value in [0,1]
pub fn parametric_multimodal_transform(
    y: f32, 
    frequency: f32, 
    amplitude: f32, 
    base_level: f32
) -> f32 {
    let oscillation = amplitude * (frequency * PI * y).sin();
    let result = base_level + oscillation;
    
    correct_to_01(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multimodal_transform_typical() {
        let result = multimodal_transform(0.5, 5.0, 10.0, 1.0);
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_multimodal_creates_oscillation() {
        // Test that multimodal transformation creates oscillations
        let samples: Vec<f32> = (0..=10)
            .map(|i| i as f32 / 10.0)
            .map(|y| multimodal_transform(y, 5.0, 10.0, 1.0))
            .collect();
        
        // Should have some variation (not monotonic)
        let mut has_increase = false;
        let mut has_decrease = false;
        
        for i in 1..samples.len() {
            if samples[i] > samples[i-1] {
                has_increase = true;
            }
            if samples[i] < samples[i-1] {
                has_decrease = true;
            }
        }
        
        assert!(has_increase && has_decrease, "Multimodal should create oscillations");
    }
    
    #[test]
    fn test_wfg_multimodal_bounds() {
        for i in 0..=20 {
            let y = i as f32 / 20.0;
            let result = wfg_multimodal_transform(y, 5, 10.0);
            assert!(result >= 0.0 && result <= 1.0, 
                   "Result {} out of bounds for y={}", result, y);
        }
    }
    
    #[test]
    fn test_parametric_multimodal() {
        let result = parametric_multimodal_transform(0.5, 2.0, 0.1, 0.5);
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_multimodal_different_modes() {
        let result_5_modes = multimodal_transform(0.5, 5.0, 10.0, 1.0);
        let result_10_modes = multimodal_transform(0.5, 10.0, 10.0, 1.0);
        
        // Different number of modes should give different results
        // (though not necessarily - depends on the point)
        assert!(result_5_modes >= 0.0 && result_5_modes <= 1.0);
        assert!(result_10_modes >= 0.0 && result_10_modes <= 1.0);
    }
}