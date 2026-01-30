//! Hyperbolic manifold implementation (Lorentz + Poincaré models).
//!
//! Based on manifold-spec.md Section 2.

use crate::helix_engine::types::ManifoldError;
use super::curvature::{LearnableCurvature, CurvatureSign};
use super::numerics::{
    EPSILON, safe_arcosh, safe_artanh, safe_div, safe_sqrt,
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
                let cosh_v = v_norm.cosh();
                let sinh_v = v_norm.sinh();
                Ok(x.iter().zip(v.iter())
                    .map(|(xi, vi)| cosh_v * xi + sinh_v * vi / v_norm)
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
                let direction: Vec<f64> = y.iter().zip(x.iter())
                    .map(|(yi, xi)| yi + kappa * inner * xi)
                    .collect();
                let dir_norm = lorentz_norm(&direction);
                Ok(direction.iter().map(|di| d * di / dir_norm).collect())
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
                Ok(v.iter().zip(x.iter())
                    .map(|(vi, xi)| vi + kappa * inner * xi)
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
}
