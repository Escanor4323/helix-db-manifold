//! Core manifold trait definition.
//!
//! Based on manifold-spec.md Sections 1-4.

use crate::helix_engine::types::ManifoldError;

/// Core trait for Riemannian manifolds.
///
/// Provides geometric operations for distance computation,
/// exponential/logarithmic maps, parallel transport, and projection.
pub trait Manifold {
    /// Intrinsic dimension of the manifold.
    fn intrinsic_dim(&self) -> usize;

    /// Ambient dimension (embedding space dimension).
    fn ambient_dim(&self) -> usize;

    /// Get current curvature value.
    fn curvature(&self) -> f64;

    /// Set curvature (for learnable curvature).
    fn set_curvature(&mut self, kappa: f64) -> Result<(), ManifoldError>;

    /// Geodesic distance between two points.
    fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, ManifoldError>;

    /// Exponential map: move from x along tangent vector v.
    fn exp_map(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError>;

    /// Logarithmic map: get tangent vector pointing from x to y.
    fn log_map(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, ManifoldError>;

    /// Project point onto manifold.
    fn project(&self, v: &[f64]) -> Result<Vec<f64>, ManifoldError>;

    /// Project vector onto tangent space at x.
    fn project_tangent(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError>;

    /// Parallel transport vector v from x to y along geodesic.
    fn parallel_transport(&self, x: &[f64], y: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError>;

    /// Get the canonical origin point.
    fn origin(&self) -> Vec<f64>;

    /// Riemannian inner product of tangent vectors at x.
    fn inner_product(&self, x: &[f64], u: &[f64], v: &[f64]) -> Result<f64, ManifoldError>;

    /// Check if point lies on manifold (within tolerance).
    fn is_on_manifold(&self, x: &[f64], tol: f64) -> bool;
}
