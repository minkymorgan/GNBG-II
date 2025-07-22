// src/cpu_reference.rs
use crate::GNBGProblem;

/// Direct port of C++ fitness evaluation for validation
pub struct CPUEvaluator {
    problem: GNBGProblem,
    pub fe_count: usize,
    pub best_found_result: f64,
    pub acceptance_reach_point: Option<usize>,
    pub fe_history: Vec<f64>,
}

impl CPUEvaluator {
    pub fn new(problem: GNBGProblem) -> Self {
        let max_evals = problem.max_evals;
        Self {
            problem,
            fe_count: 0,
            best_found_result: f64::INFINITY,
            acceptance_reach_point: None,
            fe_history: Vec::with_capacity(max_evals),
        }
    }
    
    pub fn reset(&mut self) {
        self.fe_count = 0;
        self.best_found_result = f64::INFINITY;
        self.acceptance_reach_point = None;
        self.fe_history.clear();
        self.fe_history.resize(self.problem.max_evals, 0.0);
    }

    pub fn fitness(&mut self, xvec: &[f64]) -> f64 {
        let dim = self.problem.dimension;
        let comp_num = self.problem.comp_num;
        
        let mut res = f64::INFINITY;
        let mut a = vec![0.0; dim];
        let mut temp = vec![0.0; dim];
        
        for i in 0..comp_num {
            // Step 1: Translate (x - peak position)
            for j in 0..dim {
                a[j] = xvec[j] - self.problem.comp_min_pos[i][j];
            }
            
            // Step 2: Rotate
            for j in 0..dim {
                temp[j] = 0.0;
                for k in 0..dim {
                    temp[j] += self.problem.rotation_matrices[i][j][k] * a[k];
                }
            }
            
            // Step 3: Apply asymmetric transformation
            for j in 0..dim {
                if temp[j] > 0.0 {
                    let log_val = temp[j].ln();
                    a[j] = (log_val + self.problem.mu[i][0] * 
                           (self.problem.omega[i][0] * log_val).sin() + 
                           (self.problem.omega[i][1] * log_val).sin()).exp();
                } else if temp[j] < 0.0 {
                    let log_val = (-temp[j]).ln();
                    a[j] = -(log_val + self.problem.mu[i][1] * 
                            (self.problem.omega[i][2] * log_val).sin() + 
                            (self.problem.omega[i][3] * log_val).sin()).exp();
                } else {
                    a[j] = 0.0;
                }
            }
            
            // Step 4: Compute weighted sum
            let mut fval = 0.0;
            for j in 0..dim {
                fval += a[j] * a[j] * self.problem.comp_h[i][j];
            }
            
            // Step 5: Apply final transformation
            fval = self.problem.comp_sigma[i] + fval.powf(self.problem.lambda[i]);
            
            // Take minimum
            if i == 0 {
                res = fval;
            } else {
                res = res.min(fval);
            }
        }
        
        // Update tracking
        if self.fe_count < self.problem.max_evals {
            self.fe_history.push(res);
            
            if self.fe_count == 0 {
                self.best_found_result = res;
            } else {
                self.best_found_result = self.best_found_result.min(res);
            }
            
            if res - self.problem.optimum_value < self.problem.acceptance_threshold 
                && self.acceptance_reach_point.is_none() {
                self.acceptance_reach_point = Some(self.fe_count);
            }
            
            self.fe_count += 1;
        }
        
        res
    }
}
