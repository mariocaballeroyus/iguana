/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_POISSON_POISSON_SHIFT_HPP
#define IGUANA_ELEMENT_POISSON_POISSON_SHIFT_HPP

#include <concepts>

#include <Eigen/Core>

#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Shift of a Dirichlet condition rewritten with the Poisson
 *        equation, at the points of a projection
 *
 * @details The shifted values add the weights of parametric derivatives to
 *          the values at the points from the boundary, or to their Taylor
 *          series from the surrogate, and the imposed data add to the trace
 *          g at the closest points the weights of the Cartesian derivatives
 *          of the source f. The weights take the derivatives at the closest
 *          points from the boundary, and at the points themselves from the
 *          surrogate
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
struct PoissonShift
{
    /// @brief Weights of the parametric derivatives of the field, with size
    ///        (num_points, simplex_size(m)), as
    ///        BoundaryProjection::shift_weights holds them
    Eigen::MatrixX<T> weights;

    /// @brief Weights of the Cartesian derivatives of f, with size
    ///        (num_points, simplex_size(m − 2)), and no columns below m = 2
    Eigen::MatrixX<T> source_weights;
};

/**
 * @brief Shift of a Dirichlet condition rewritten with the Poisson equation,
 *        from the series of the solution along the lines between the points
 *        and their closest points
 *
 *     T^m u + ρ^m(u) = g(x_Γ),    t = |x̃ − x_Γ|,    e = (x̃ − x_Γ) / t
 *     T^m u = u(x̃) − Σ_{k=1}^{m} t^k / k! ∂_e^k u(x_Γ)
 *     ρ^m(u) = Σ_{k=2}^{m} t^k / k! D_k (Δu + f)(x_Γ)
 *     D_k = Σ_{i=0}^{⌊k/2⌋−1} (−1)^i ∂_⊥^{2i} ∂_e^{k−2−2i}
 *     ∂_e^k − D_k Δ = (−1)^⌊k/2⌋ ∂_⊥^{2⌊k/2⌋} ∂_e^{k−2⌊k/2⌋}
 *     S^{m,r}_d u(x̃) = g(x_Γ),    d = x_Γ − x̃
 *     S^{m,r}_d u = S^m_d u − t² Σ_{k=2}^{m} D_d^{k−2} (Δu + f) / k!
 *     D_d^j = Σ_{i=0}^{⌊j/2⌋} (−1)^i (d^⊥ · ∇)^{2i} (d · ∇)^{j−2i}
 *
 * @details From the boundary, the condition imposes at each point x̃ the
 *          series T^m of the solution from its closest point x_Γ, plus the
 *          terms ρ^m of the residual of −Δu = f there, which vanish on the
 *          solution, with ∂_⊥ along e turned by +90°. Together they take
 *          every derivative along e from the second on from the equation:
 *          the shifted values hold u at the point and the derivatives of u
 *          at x_Γ with at most one along e, and the data g and the
 *          derivatives of f at x_Γ. From the surrogate, it imposes the
 *          Taylor series S^m_d of the solution from x̃ less the terms of the
 *          residual there, with d^⊥ the vector d turned by +90°, so that the
 *          derivatives of u and of f are those at x̃. The source must then
 *          extend smoothly beyond the domain, to the points outside it. No
 *          normal or curvature of the boundary enters, so the closest
 *          points may lie at corners
 *
 * @tparam Basis Basis type of the patch
 *
 * @param patch Patch whose parameter space holds the points
 * @param projection Closest points of the points on a boundary
 * @param order Highest order m of the series
 * @param center Point the series expands about
 * @return The weights of the shift at each point of @p projection
 *
 * @throws std::invalid_argument If @p order is negative or if the map of
 *         @p patch is not affine
 *
 * @pre @p projection was computed through @p patch
 */
template<typename Basis> requires (Basis::dimension == 2)
PoissonShift<typename Basis::Scalar> poisson_shift(
    const Patch<Basis, 2>& patch,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order,
    ShiftCenter center = ShiftCenter::boundary);

} // namespace iguana

#endif // IGUANA_ELEMENT_POISSON_POISSON_SHIFT_HPP
