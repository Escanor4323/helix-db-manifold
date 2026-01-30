//! Product manifold implementation.
//!
//! Based on manifold-spec.md Section 4.

use crate::helix_engine::types::ManifoldError;
use super::euclidean::EuclideanManifold;
use super::hyperbolic::HyperbolicManifold;
use super::spherical::SphericalManifold;
use super::numerics::safe_sqrt;
use super::traits::Manifold;

/// Product manifold M = H^d1 × S^d2 × E^d3.
///
/// Points are represented as concatenated vectors:
/// [hyperbolic_coords | spherical_coords | euclidean_coords]
#[derive(Debug, Clone)]
pub struct ProductManifold {
    hyperbolic: HyperbolicManifold,
    spherical: SphericalManifold,
    euclidean: EuclideanManifold,
    h_ambient_dim: usize,
    s_ambient_dim: usize,
    e_dim: usize,
}

impl ProductManifold {
    pub fn new(h_dim: usize, s_dim: usize, e_dim: usize) -> Self {
        let hyperbolic = HyperbolicManifold::new(h_dim);
        let spherical = SphericalManifold::new(s_dim);
        let euclidean = EuclideanManifold::new(e_dim);
        Self {
            h_ambient_dim: hyperbolic.ambient_dim(),
            s_ambient_dim: spherical.ambient_dim(),
            e_dim,
            hyperbolic,
            spherical,
            euclidean,
        }
    }

    pub fn with_curvatures(
        h_dim: usize, h_kappa: f64,
        s_dim: usize, s_kappa: f64,
        e_dim: usize,
    ) -> Result<Self, ManifoldError> {
        let hyperbolic = HyperbolicManifold::with_curvature(h_dim, h_kappa)?;
        let spherical = SphericalManifold::with_curvature(s_dim, s_kappa)?;
        let euclidean = EuclideanManifold::new(e_dim);
        Ok(Self {
            h_ambient_dim: hyperbolic.ambient_dim(),
            s_ambient_dim: spherical.ambient_dim(),
            e_dim,
            hyperbolic,
            spherical,
            euclidean,
        })
    }

    /// Split point into component slices.
    fn split<'a>(&self, x: &'a [f64]) -> Result<(&'a [f64], &'a [f64], &'a [f64]), ManifoldError> {
        if x.len() != self.ambient_dim() {
            return Err(ManifoldError::DimensionMismatch);
        }
        let h_end = self.h_ambient_dim;
        let s_end = h_end + self.s_ambient_dim;
        Ok((&x[..h_end], &x[h_end..s_end], &x[s_end..]))
    }

    /// Combine component vectors into product point.
    fn combine(&self, h: Vec<f64>, s: Vec<f64>, e: Vec<f64>) -> Vec<f64> {
        let mut result = h;
        result.extend(s);
        result.extend(e);
        result
    }

    pub fn hyperbolic_component(&self) -> &HyperbolicManifold {
        &self.hyperbolic
    }

    pub fn spherical_component(&self) -> &SphericalManifold {
        &self.spherical
    }

    pub fn euclidean_component(&self) -> &EuclideanManifold {
        &self.euclidean
    }
}

impl Manifold for ProductManifold {
    fn intrinsic_dim(&self) -> usize {
        self.hyperbolic.intrinsic_dim() + self.spherical.intrinsic_dim() + self.euclidean.intrinsic_dim()
    }

    fn ambient_dim(&self) -> usize {
        self.h_ambient_dim + self.s_ambient_dim + self.e_dim
    }

    fn curvature(&self) -> f64 {
        0.0
    }

    fn set_curvature(&mut self, _kappa: f64) -> Result<(), ManifoldError> {
        Err(ManifoldError::InvalidCurvature(
            "ProductManifold has component-dependent curvature".to_string()
        ))
    }

    /// d(x,y)² = d_H² + d_S² + d_E²
    fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, ManifoldError> {
        let (xh, xs, xe) = self.split(x)?;
        let (yh, ys, ye) = self.split(y)?;
        
        let dh = self.hyperbolic.distance(xh, yh)?;
        let ds = self.spherical.distance(xs, ys)?;
        let de = self.euclidean.distance(xe, ye)?;
        
        Ok(safe_sqrt(dh * dh + ds * ds + de * de))
    }

    fn exp_map(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let (xh, xs, xe) = self.split(x)?;
        let (vh, vs, ve) = self.split(v)?;
        
        let rh = self.hyperbolic.exp_map(xh, vh)?;
        let rs = self.spherical.exp_map(xs, vs)?;
        let re = self.euclidean.exp_map(xe, ve)?;
        
        Ok(self.combine(rh, rs, re))
    }

    fn log_map(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let (xh, xs, xe) = self.split(x)?;
        let (yh, ys, ye) = self.split(y)?;
        
        let vh = self.hyperbolic.log_map(xh, yh)?;
        let vs = self.spherical.log_map(xs, ys)?;
        let ve = self.euclidean.log_map(xe, ye)?;
        
        Ok(self.combine(vh, vs, ve))
    }

    fn project(&self, v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let (vh, vs, ve) = self.split(v)?;
        
        let rh = self.hyperbolic.project(vh)?;
        let rs = self.spherical.project(vs)?;
        let re = self.euclidean.project(ve)?;
        
        Ok(self.combine(rh, rs, re))
    }

    fn project_tangent(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let (xh, xs, xe) = self.split(x)?;
        let (vh, vs, ve) = self.split(v)?;
        
        let rh = self.hyperbolic.project_tangent(xh, vh)?;
        let rs = self.spherical.project_tangent(xs, vs)?;
        let re = self.euclidean.project_tangent(xe, ve)?;
        
        Ok(self.combine(rh, rs, re))
    }

    fn parallel_transport(&self, x: &[f64], y: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        let (xh, xs, xe) = self.split(x)?;
        let (yh, ys, ye) = self.split(y)?;
        let (vh, vs, ve) = self.split(v)?;
        
        let rh = self.hyperbolic.parallel_transport(xh, yh, vh)?;
        let rs = self.spherical.parallel_transport(xs, ys, vs)?;
        let re = self.euclidean.parallel_transport(xe, ye, ve)?;
        
        Ok(self.combine(rh, rs, re))
    }

    fn origin(&self) -> Vec<f64> {
        self.combine(
            self.hyperbolic.origin(),
            self.spherical.origin(),
            self.euclidean.origin(),
        )
    }

    fn inner_product(&self, x: &[f64], u: &[f64], v: &[f64]) -> Result<f64, ManifoldError> {
        let (xh, xs, xe) = self.split(x)?;
        let (uh, us, ue) = self.split(u)?;
        let (vh, vs, ve) = self.split(v)?;
        
        let ih = self.hyperbolic.inner_product(xh, uh, vh)?;
        let is = self.spherical.inner_product(xs, us, vs)?;
        let ie = self.euclidean.inner_product(xe, ue, ve)?;
        
        Ok(ih + is + ie)
    }

    fn is_on_manifold(&self, x: &[f64], tol: f64) -> bool {
        if let Ok((xh, xs, xe)) = self.split(x) {
            self.hyperbolic.is_on_manifold(xh, tol)
                && self.spherical.is_on_manifold(xs, tol)
                && self.euclidean.is_on_manifold(xe, tol)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_origin_on_manifold() {
        let m = ProductManifold::new(2, 2, 3);
        let o = m.origin();
        assert!(m.is_on_manifold(&o, 1e-10));
    }

    #[test]
    fn test_product_distance_l2() {
        let m = ProductManifold::new(2, 2, 2);
        let o = m.origin();
        let d = m.distance(&o, &o).unwrap();
        assert!(d < 1e-6, "expected small distance, got {}", d);
    }

    #[test]
    fn test_product_dimensions() {
        let m = ProductManifold::new(2, 3, 4);
        assert_eq!(m.intrinsic_dim(), 2 + 3 + 4);
        assert_eq!(m.ambient_dim(), 3 + 4 + 4);
    }
}
