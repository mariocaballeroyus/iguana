/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_POISSON_POISSON_SHIFT_HPP
#define IGUANA_ELEMENT_POISSON_POISSON_SHIFT_HPP

#include <concepts>

#include <Eigen/Core>

#include "iguana/geometry/boundary.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Shift of a Dirichlet condition rewritten with the Poisson
 *        equation, at the points of a projection
 *
 * @details The shifted values combine the values at the points with the
 *          weights of parametric derivatives at their closest points, and
 *          the imposed data the weights of the Cartesian derivatives of the
 *          trace g and of the source f there
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
struct PoissonShift
{
    /// @brief Highest total order of the parametric derivatives the
    ///        weights combine
    int order;

    /// @brief Signed distance t of each point from its closest point, along
    ///        the outward normal, positive outside the domain
    Eigen::VectorX<T> distances;

    /// @brief Weights of the parametric derivatives of the field at the
    ///        closest points, with size (num_points, simplex_size(order)),
    ///        as BoundaryProjection::shift_weights holds them
    Eigen::MatrixX<T> weights;

    /// @brief Weights of the Cartesian derivatives of g at the closest
    ///        points, with size (num_points, simplex_size(m))
    Eigen::MatrixX<T> value_weights;

    /// @brief Weights of the Cartesian derivatives of f at the closest
    ///        points, with size (num_points, simplex_size(m − 2)), and no
    ///        columns below m = 2
    Eigen::MatrixX<T> source_weights;
};

/**
 * @brief Shift of a Dirichlet condition rewritten with the Poisson equation,
 *        from the normal series of the solution at the closest points
 *
 *     Ŝ u = u(x̃) − Σ_{k=1}^{m} t^k / k! U_k^lin,
 *     G = Σ_{k=0}^{m} t^k / k! U_k^data,    t = (x̃ − x_Γ) · ν
 *
 * @details The condition imposes Ŝ u = G at each point x̃. U_k^lin holds the
 *          normal derivative of u at the closest point x_Γ and its
 *          derivatives along the boundary, and U_k^data the trace g, the
 *          source f and the curvature, as poisson_normal_series() gives
 *          them. The boundary must be smooth at the closest points, where
 *          the distance vectors lie along the normals
 *
 * @tparam Basis Basis type of the patch
 *
 * @param patch Patch whose parameter space holds the points
 * @param boundary Boundary of curves the points are projected onto
 * @param projection Closest points on @p boundary
 * @param order Highest order m of the series
 * @return The weights of the shift at each point of @p projection
 *
 * @throws std::invalid_argument If @p order is negative, if the map of
 *         @p patch is not affine, or if a distance vector does not lie
 *         along the normal at its closest point, as at a corner
 *
 * @pre @p projection was computed onto @p boundary through @p patch
 */
template<typename Basis> requires (Basis::dimension == 2)
PoissonShift<typename Basis::Scalar> poisson_shift(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order);

} // namespace iguana

#endif // IGUANA_ELEMENT_POISSON_POISSON_SHIFT_HPP
