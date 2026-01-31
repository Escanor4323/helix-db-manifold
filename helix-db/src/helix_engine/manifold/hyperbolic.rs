//! Hyperbolic manifold implementation (Lorentz + Poincaré models).
//!
//! Based on manifold-spec.md Section 2.

use crate::helix_engine::types::ManifoldError;
use super::curvature::{LearnableCurvature, CurvatureSign};
use super::numerics::{
    EPSILON, safe_arcosh, safe_artanh, safe_div, safe_sqrt,
    safe_cosh, safe_sinh,
    dot, norm, norm_squared, minkowski_dot, lorentz_norm,
};
use super::traits::Manifold;

/// Which hyperbolic model to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperbolicModel {
    Lorentz,
    Poincare,
}

/// Hyperbolic manifold with learnable curvature.
#[derive(Debug, Clone)]
pub struct HyperbolicManifold {
    dim: usize,
    curvature: LearnableCurvature,
    model: HyperbolicModel,
}

impl HyperbolicManifold {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            curvature: LearnableCurvature::new(-1.0, CurvatureSign::Negative),
            model: HyperbolicModel::Lorentz,
        }
    }

    pub fn with_curvature(dim: usize, kappa: f64) -> Result<Self, ManifoldError> {
        if kappa >= 0.0 {
            return Err(ManifoldError::InvalidCurvature(
                "Hyperbolic curvature must be negative".to_string()
            ));
        }
        Ok(Self {
            dim,
            curvature: LearnableCurvature::new(kappa, CurvatureSign::Negative),
            model: HyperbolicModel::Lorentz,
        })
    }

    pub fn with_model(mut self, model: HyperbolicModel) -> Self {
        self.model = model;
        self
    }

    pub fn model(&self) -> HyperbolicModel {
        self.model
    }

    /// Convert Lorentz point to Poincaré ball.
    pub fn lorentz_to_poincare(&self, x: &[f64]) -> Vec<f64> {
        let kappa = self.curvature.kappa();
        let scale = x[0] + 1.0 / (-kappa).sqrt();
        x[1..].iter().map(|xi| xi / scale).collect()
    }

    /// Convert Poincaré ball point to Lorentz.
    pub fn poincare_to_lorentz(&self, p: &[f64]) -> Vec<f64> {
        let kappa = self.curvature.kappa();
        let p_norm_sq = norm_squared(p);
        let denom = 1.0 + kappa * p_norm_sq;
        let x0 = (1.0 - kappa * p_norm_sq) / (denom * (-kappa).sqrt());
        let mut result = vec![x0];
        for pi in p {
            result.push(2.0 * pi / denom);
        }
        result
    }

    /// Conformal factor for Poincaré model.
    fn conformal_factor(&self, x: &[f64]) -> f64 {
        let kappa = self.curvature.kappa();
        2.0 / (1.0 + kappa * norm_squared(x))
    }

    /// Möbius addition: x ⊕ y
    fn mobius_add(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        let kappa = self.curvature.kappa();
        let xy = dot(x, y);
        let x_sq = norm_squared(x);
        let y_sq = norm_squared(y);
        
        let num_x = 1.0 - 2.0 * kappa * xy - kappa * y_sq;
        let num_y = 1.0 + kappa * x_sq;
        let denom = 1.0 - 2.0 * kappa * xy + kappa * kappa * x_sq * y_sq;
        
        x.iter().zip(y.iter())
            .map(|(xi, yi)| safe_div(num_x * xi + num_y * yi, denom))
            .collect()
    }

    /// Check if vector v is tangent to the hyperboloid at point x.
    /// A tangent vector satisfies ⟨v, x⟩_L = 0 within tolerance.
    /// Only valid for Lorentz model.
    pub fn is_tangent(&self, x: &[f64], v: &[f64], tol: f64) -> bool {
        if self.model != HyperbolicModel::Lorentz {
            return true; // Poincaré has no constraint
        }
        if x.len() != self.dim + 1 || v.len() != self.dim + 1 {
            return false;
        }
        let inner = minkowski_dot(v, x);
        inner.abs() < tol
    }
}

impl Manifold for HyperbolicManifold {
    fn intrinsic_dim(&self) -> usize {
        self.dim
    }

    fn ambient_dim(&self) -> usize {
        match self.model {
            HyperbolicModel::Lorentz => self.dim + 1,
            HyperbolicModel::Poincare => self.dim,
        }
    }

    fn curvature(&self) -> f64 {
        self.curvature.kappa()
    }

    fn set_curvature(&mut self, kappa: f64) -> Result<(), ManifoldError> {
        if kappa >= 0.0 {
            return Err(ManifoldError::InvalidCurvature(
                "Hyperbolic curvature must be negative".to_string()
            ));
        }
        let theta = inverse_softplus_for(-kappa);
        self.curvature.set_theta(theta);
        Ok(())
    }

    fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, ManifoldError> {
        let kappa = self.curvature.kappa();
        
        match self.model {
            HyperbolicModel::Lorentz => {
                if x.len() != self.dim + 1 || y.len() != self.dim + 1 {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let inner = minkowski_dot(x, y);
                let arg = kappa * inner;
                if arg <= 1.0 + EPSILON {
                    return Ok(0.0);
                }
                Ok(safe_arcosh(arg) / (-kappa).sqrt())
            }
            HyperbolicModel::Poincare => {
                if x.len() != self.dim || y.len() != self.dim {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let diff: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
                let diff_sq = norm_squared(&diff);
                if diff_sq < EPSILON {
                    return Ok(0.0);
                }
                let x_sq = norm_squared(x);
                let y_sq = norm_squared(y);
                let num = 2.0 * (-kappa) * diff_sq;
                let denom = (1.0 + kappa * x_sq) * (1.0 + kappa * y_sq);
                let arg = 1.0 + safe_div(num, denom);
                Ok(safe_arcosh(arg) / (-kappa).sqrt())
            }
        }
    }

    fn exp_map(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let kappa = self.curvature.kappa();
        
        match self.model {
            HyperbolicModel::Lorentz => {
                if x.len() != self.dim + 1 || v.len() != self.dim + 1 {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let v_norm = lorentz_norm(v);
                if v_norm < EPSILON {
                    return Ok(x.to_vec());
                }
                // Use safe_cosh/safe_sinh to prevent overflow in deep networks
                let cosh_v = safe_cosh(v_norm);
                let sinh_v = safe_sinh(v_norm);
                let scale = safe_div(sinh_v, v_norm);
                Ok(x.iter().zip(v.iter())
                    .map(|(xi, vi)| cosh_v * xi + scale * vi)
                    .collect())
            }
            HyperbolicModel::Poincare => {
                if x.len() != self.dim || v.len() != self.dim {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let v_norm = norm(v);
                if v_norm < EPSILON {
                    return Ok(x.to_vec());
                }
                let lambda = self.conformal_factor(x);
                let arg = (-kappa).sqrt() * lambda * v_norm / 2.0;
                let scale = arg.tanh() / ((-kappa).sqrt() * v_norm);
                let scaled_v: Vec<f64> = v.iter().map(|vi| vi * scale).collect();
                Ok(self.mobius_add(x, &scaled_v))
            }
        }
    }

    fn log_map(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let kappa = self.curvature.kappa();
        
        match self.model {
            HyperbolicModel::Lorentz => {
                if x.len() != self.dim + 1 || y.len() != self.dim + 1 {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let d = self.distance(x, y)?;
                if d < EPSILON {
                    return Ok(vec![0.0; self.dim + 1]);
                }
                let inner = minkowski_dot(x, y);
                // Correct direction: y - κ⟨x,y⟩_L x lies in tangent space at x
                let direction: Vec<f64> = y.iter().zip(x.iter())
                    .map(|(yi, xi)| yi - kappa * inner * xi)
                    .collect();
                let dir_norm = lorentz_norm(&direction);
                // Use safe_div to prevent division by zero in extreme cases
                let scale = safe_div(d, dir_norm);
                Ok(direction.iter().map(|di| scale * di).collect())
            }
            HyperbolicModel::Poincare => {
                if x.len() != self.dim || y.len() != self.dim {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let neg_x: Vec<f64> = x.iter().map(|xi| -xi).collect();
                let diff = self.mobius_add(&neg_x, y);
                let diff_norm = norm(&diff);
                if diff_norm < EPSILON {
                    return Ok(vec![0.0; self.dim]);
                }
                let lambda = self.conformal_factor(x);
                let scale = 2.0 / ((-kappa).sqrt() * lambda) * safe_artanh((-kappa).sqrt() * diff_norm);
                Ok(diff.iter().map(|di| scale * di / diff_norm).collect())
            }
        }
    }

    fn project(&self, v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let kappa = self.curvature.kappa();
        
        match self.model {
            HyperbolicModel::Lorentz => {
                if v.len() != self.dim + 1 {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let spatial: Vec<f64> = v[1..].to_vec();
                let spatial_sq: f64 = norm_squared(&spatial);
                let x0 = safe_sqrt(1.0 / (-kappa) + spatial_sq);
                let mut result = vec![x0];
                result.extend(spatial);
                Ok(result)
            }
            HyperbolicModel::Poincare => {
                if v.len() != self.dim {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let max_norm = 1.0 / (-kappa).sqrt() - EPSILON;
                let v_norm = norm(v);
                if v_norm >= max_norm {
                    Ok(v.iter().map(|vi| vi * max_norm / v_norm).collect())
                } else {
                    Ok(v.to_vec())
                }
            }
        }
    }

    fn project_tangent(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let kappa = self.curvature.kappa();
        
        match self.model {
            HyperbolicModel::Lorentz => {
                if x.len() != self.dim + 1 || v.len() != self.dim + 1 {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let inner = minkowski_dot(v, x);
                // Correct formula: v - κ⟨v,x⟩_L x gives ⟨proj,x⟩_L = 0
                // Note: spec says + but that's incorrect for the hyperboloid constraint
                Ok(v.iter().zip(x.iter())
                    .map(|(vi, xi)| vi - kappa * inner * xi)
                    .collect())
            }
            HyperbolicModel::Poincare => {
                Ok(v.to_vec())
            }
        }
    }

    fn parallel_transport(&self, x: &[f64], y: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let kappa = self.curvature.kappa();
        
        match self.model {
            HyperbolicModel::Lorentz => {
                if x.len() != self.dim + 1 || y.len() != self.dim + 1 || v.len() != self.dim + 1 {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let inner_xy = minkowski_dot(x, y);
                let inner_yv = minkowski_dot(y, v);
                let denom = 1.0 - kappa * inner_xy;
                let coeff = kappa * inner_yv / denom;
                Ok(v.iter().zip(x.iter().zip(y.iter()))
                    .map(|(vi, (xi, yi))| vi - coeff * (xi + yi))
                    .collect())
            }
            HyperbolicModel::Poincare => {
                let lambda_x = self.conformal_factor(x);
                let lambda_y = self.conformal_factor(y);
                Ok(v.iter().map(|vi| vi * lambda_x / lambda_y).collect())
            }
        }
    }

    fn origin(&self) -> Vec<f64> {
        let kappa = self.curvature.kappa();
        match self.model {
            HyperbolicModel::Lorentz => {
                let mut o = vec![1.0 / (-kappa).sqrt()];
                o.extend(vec![0.0; self.dim]);
                o
            }
            HyperbolicModel::Poincare => {
                vec![0.0; self.dim]
            }
        }
    }

    fn inner_product(&self, x: &[f64], u: &[f64], v: &[f64]) -> Result<f64, ManifoldError> {
        match self.model {
            HyperbolicModel::Lorentz => {
                if x.len() != self.dim + 1 || u.len() != self.dim + 1 || v.len() != self.dim + 1 {
                    return Err(ManifoldError::DimensionMismatch);
                }
                Ok(minkowski_dot(u, v))
            }
            HyperbolicModel::Poincare => {
                if x.len() != self.dim || u.len() != self.dim || v.len() != self.dim {
                    return Err(ManifoldError::DimensionMismatch);
                }
                let lambda = self.conformal_factor(x);
                Ok(lambda * lambda * dot(u, v))
            }
        }
    }

    fn is_on_manifold(&self, x: &[f64], tol: f64) -> bool {
        let kappa = self.curvature.kappa();
        match self.model {
            HyperbolicModel::Lorentz => {
                if x.len() != self.dim + 1 || x[0] <= 0.0 {
                    return false;
                }
                let constraint = minkowski_dot(x, x) - 1.0 / kappa;
                constraint.abs() < tol
            }
            HyperbolicModel::Poincare => {
                if x.len() != self.dim {
                    return false;
                }
                let max_norm_sq = 1.0 / (-kappa);
                norm_squared(x) < max_norm_sq
            }
        }
    }
}

fn inverse_softplus_for(k: f64) -> f64 {
    if k > 20.0 { k } else { (k.exp() - 1.0).max(1e-10).ln() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorentz_origin_on_manifold() {
        let m = HyperbolicManifold::new(2);
        let o = m.origin();
        assert!(m.is_on_manifold(&o, 1e-10));
    }

    #[test]
    fn test_poincare_origin_on_manifold() {
        let m = HyperbolicManifold::new(2).with_model(HyperbolicModel::Poincare);
        let o = m.origin();
        assert!(m.is_on_manifold(&o, 1e-10));
    }

    #[test]
    fn test_lorentz_poincare_conversion() {
        let m = HyperbolicManifold::new(2);
        let o = m.origin();
        let p = m.lorentz_to_poincare(&o);
        let o2 = m.poincare_to_lorentz(&p);
        for (a, b) in o.iter().zip(o2.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
    }

    // === Issue #2: Fuzz tests with extreme values ===

    #[test]
    fn test_lorentz_exp_map_extreme_tangent_norm() {
        let m = HyperbolicManifold::new(3);
        let origin = m.origin();
        
        // Very large tangent vector (should not overflow)
        let huge_v = vec![0.0, 1e10, 1e10, 1e10];
        let result = m.exp_map(&origin, &huge_v);
        assert!(result.is_ok());
        let point = result.unwrap();
        for val in &point {
            assert!(!val.is_nan(), "NaN detected in exp_map with huge tangent");
            assert!(val.is_finite(), "Infinity detected in exp_map with huge tangent");
        }
        
        // Very tiny tangent vector
        let tiny_v = vec![0.0, 1e-15, 1e-15, 1e-15];
        let result = m.exp_map(&origin, &tiny_v);
        assert!(result.is_ok());
        let point = result.unwrap();
        for val in &point {
            assert!(!val.is_nan(), "NaN detected in exp_map with tiny tangent");
        }
    }

    #[test]
    fn test_lorentz_log_map_no_nan() {
        let m = HyperbolicManifold::new(3);
        let origin = m.origin();
        
        // Create a point on the manifold via projection
        let raw = vec![2.0, 0.5, 0.3, 0.1];
        let point = m.project(&raw).unwrap();
        
        let result = m.log_map(&origin, &point);
        assert!(result.is_ok());
        let tangent = result.unwrap();
        for val in &tangent {
            assert!(!val.is_nan(), "NaN detected in log_map");
            assert!(val.is_finite(), "Infinity detected in log_map");
        }
    }

    #[test]
    fn test_lorentz_distance_no_nan() {
        let m = HyperbolicManifold::new(3);
        let origin = m.origin();
        
        // Same point distance
        let d = m.distance(&origin, &origin).unwrap();
        assert!(!d.is_nan());
        assert!(d >= 0.0);
        
        // Distant points
        let raw = vec![10.0, 5.0, 3.0, 1.0];
        let point = m.project(&raw).unwrap();
        let d = m.distance(&origin, &point).unwrap();
        assert!(!d.is_nan());
        assert!(d.is_finite());
        assert!(d > 0.0);
    }

    // === Issue #2: Constraint test for proj ===

    #[test]
    fn test_tangent_projection_constraint() {
        let m = HyperbolicManifold::new(3);
        let origin = m.origin();
        
        // Arbitrary vector (not necessarily tangent)
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let projected = m.project_tangent(&origin, &v).unwrap();
        
        // Calculate inner product for debugging
        let inner = minkowski_dot(&projected, &origin);
        
        // Use looser tolerance for floating-point operations
        assert!(inner.abs() < 1e-5,
            "Tangent projection failed: inner product = {}, expected ≈ 0", inner);
    }

    #[test]
    fn test_is_tangent_validation() {
        let m = HyperbolicManifold::new(2);
        let origin = m.origin(); // [1, 0, 0] for κ=-1
        
        // Valid tangent: spatial components only
        let valid_tangent = vec![0.0, 1.0, 0.0];
        assert!(m.is_tangent(&origin, &valid_tangent, 1e-10));
        
        // Invalid tangent: has time component
        let invalid_tangent = vec![1.0, 0.0, 0.0];
        assert!(!m.is_tangent(&origin, &invalid_tangent, 1e-10));
    }

    // === Issue #2: Stress test ensuring no panics ===

    #[test]
    fn test_stress_mixed_operations() {
        let m = HyperbolicManifold::new(5);
        let origin = m.origin();
        
        // Chain of operations  
        for i in 1..100 {
            let scale = (i as f64) * 0.1;
            let v = vec![0.0, scale, scale * 0.5, scale * 0.3, scale * 0.1, scale * 0.05];
            
            let point = m.exp_map(&origin, &v).unwrap();
            let _ = m.distance(&origin, &point).unwrap();
            let v_back = m.log_map(&origin, &point).unwrap();
            
            // Verify no NaN or infinity in results
            for val in &v_back {
                assert!(!val.is_nan(), "NaN in log_map at iteration {}", i);
                assert!(val.is_finite(), "Infinity in log_map at iteration {}", i);
            }
            
            // Verify tangent constraint with appropriate tolerance
            let inner = minkowski_dot(&v_back, &origin);
            assert!(inner.abs() < 1e-5,
                "Log map output not tangent at iteration {}: inner = {}", i, inner);
        }
    }
}

