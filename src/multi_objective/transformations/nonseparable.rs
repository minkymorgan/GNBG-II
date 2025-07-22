/// Non-separable transformation functions
/// 
/// Creates variable dependencies that cannot be optimized independently

use super::correct_to_01;

/// Non-separable reduction transformation
/// 
/// Groups variables together and creates dependencies between them.
/// This transformation requires processing multiple variables simultaneously.
/// 
/// # Arguments
/// * `y_vector` - Input vector of values in [0,1]
/// * `A` - Reduction parameter (number of variables to group)
/// 
/// # Returns
/// Vector of transformed values
pub fn nonseparable_reduction(y_vector: &[f32], A: u32) -> Vec<f32> {
    let n = y_vector.len();
    let A = (A as usize).min(n);
    
    if A == 0 {
        return y_vector.to_vec();
    }
    
    let mut result = Vec::with_capacity(n);
    
    // Process variables in groups of size A
    for i in 0..n {
        let start_idx = (i / A) * A;
        let end_idx = (start_idx + A).min(n);
        
        // Sum variables in the group
        let group_sum: f32 = y_vector[start_idx..end_idx].iter().sum();
        let group_size = end_idx - start_idx;
        
        // Apply non-separable transformation
        let transformed = group_sum / (group_size as f32);
        result.push(correct_to_01(transformed));
    }
    
    result
}

/// WFG-style non-separable transformation
/// 
/// Implements the specific non-separable transformation from WFG
/// 
/// # Arguments
/// * `y_vector` - Input vector
/// * `A` - Number of variables in each group
/// 
/// # Returns
/// Transformed vector
pub fn wfg_nonseparable_reduction(y_vector: &[f32], A: u32) -> Vec<f32> {
    let n = y_vector.len();
    let A = A as usize;
    
    if A >= n {
        // If A >= n, just return weighted average
        let sum: f32 = y_vector.iter().sum();
        let avg = sum / (n as f32);
        return vec![correct_to_01(avg); n];
    }
    
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        // Compute weighted sum over the group
        for j in 0..A {
            let idx = (i + j) % n;
            let weight = 1.0; // Can be adjusted for different weighting schemes
            
            numerator += weight * y_vector[idx];
            denominator += weight;
        }
        
        let transformed = if denominator > 0.0 {
            numerator / denominator
        } else {
            y_vector[i]
        };
        
        result.push(correct_to_01(transformed));
    }
    
    result
}

/// Weighted non-separable transformation
/// 
/// Allows custom weights for variable interactions
/// 
/// # Arguments
/// * `y_vector` - Input vector
/// * `weights` - Weights for each variable
/// * `group_size` - Size of variable groups
/// 
/// # Returns
/// Transformed vector
pub fn weighted_nonseparable_reduction(
    y_vector: &[f32], 
    weights: &[f32], 
    group_size: usize
) -> Vec<f32> {
    let n = y_vector.len();
    
    if weights.len() != n {
        // Fallback to uniform weights
        return nonseparable_reduction(y_vector, group_size as u32);
    }
    
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        // Process group around position i
        for j in 0..group_size {
            let idx = (i + j) % n;
            weighted_sum += weights[idx] * y_vector[idx];
            weight_sum += weights[idx];
        }
        
        let transformed = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            y_vector[i]
        };
        
        result.push(correct_to_01(transformed));
    }
    
    result
}

/// Linear non-separable transformation
/// 
/// Creates linear dependencies between variables
/// 
/// # Arguments
/// * `y_vector` - Input vector
/// * `interaction_matrix` - Matrix defining variable interactions
/// 
/// # Returns
/// Transformed vector
pub fn linear_nonseparable_transform(
    y_vector: &[f32], 
    interaction_matrix: &[Vec<f32>]
) -> Vec<f32> {
    let n = y_vector.len();
    
    if interaction_matrix.len() != n {
        return y_vector.to_vec();
    }
    
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        if interaction_matrix[i].len() != n {
            result.push(y_vector[i]);
            continue;
        }
        
        let mut transformed = 0.0;
        for j in 0..n {
            transformed += interaction_matrix[i][j] * y_vector[j];
        }
        
        result.push(correct_to_01(transformed));
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nonseparable_reduction_basic() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let result = nonseparable_reduction(&input, 2);
        
        assert_eq!(result.len(), input.len());
        
        // All values should be in [0,1]
        for &val in &result {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
    
    #[test]
    fn test_nonseparable_creates_dependencies() {
        let input1 = vec![0.1, 0.2, 0.3, 0.4];
        let input2 = vec![0.1, 0.9, 0.3, 0.4]; // Changed one value
        
        let result1 = nonseparable_reduction(&input1, 2);
        let result2 = nonseparable_reduction(&input2, 2);
        
        // Results should be different and not just at the changed position
        // (due to non-separability)
        let differences: Vec<bool> = result1.iter()
            .zip(result2.iter())
            .map(|(a, b)| (a - b).abs() > 1e-6)
            .collect();
        
        let changed_positions = differences.iter().filter(|&&x| x).count();
        assert!(changed_positions > 1, "Non-separable should affect multiple positions");
    }
    
    #[test]
    fn test_wfg_nonseparable_reduction() {
        let input = vec![0.2, 0.4, 0.6, 0.8];
        let result = wfg_nonseparable_reduction(&input, 2);
        
        assert_eq!(result.len(), input.len());
        
        for &val in &result {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
    
    #[test]
    fn test_weighted_nonseparable() {
        let input = vec![0.2, 0.4, 0.6, 0.8];
        let weights = vec![1.0, 2.0, 1.0, 2.0];
        let result = weighted_nonseparable_reduction(&input, &weights, 2);
        
        assert_eq!(result.len(), input.len());
        
        for &val in &result {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
    
    #[test]
    fn test_linear_nonseparable() {
        let input = vec![0.5, 0.5];
        let matrix = vec![
            vec![0.8, 0.2],
            vec![0.3, 0.7],
        ];
        
        let result = linear_nonseparable_transform(&input, &matrix);
        
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.5).abs() < 1e-6); // 0.8*0.5 + 0.2*0.5 = 0.5
        assert!((result[1] - 0.5).abs() < 1e-6); // 0.3*0.5 + 0.7*0.5 = 0.5
    }
    
    #[test]
    fn test_nonseparable_empty_input() {
        let input = vec![];
        let result = nonseparable_reduction(&input, 2);
        assert!(result.is_empty());
    }
    
    #[test]
    fn test_nonseparable_large_A() {
        let input = vec![0.1, 0.2, 0.3];
        let result = nonseparable_reduction(&input, 10); // A > n
        
        assert_eq!(result.len(), input.len());
        
        // Should still produce valid results
        for &val in &result {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
}