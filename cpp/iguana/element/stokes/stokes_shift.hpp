/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_STOKES_STOKES_SHIFT_HPP
#define IGUANA_ELEMENT_STOKES_STOKES_SHIFT_HPP

#include <concepts>
#include <vector>

#include <Eigen/Core>

#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Shift of the velocity of a Dirichlet condition rewritten with the
 *        Stokes equation, at the points of a projection
 *
 * @details The shifted value of each velocity component adds to its value
 *          at the point the weights of its parametric derivatives at the
 *          closest point, the same for every component, and those of the
 *          parametric derivatives of the pressure there, of its own
 *          component. The imposed data add to the component of the trace g
 *          the weights of the Cartesian derivatives of that of the body
 *          force f
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
struct StokesShift
{
    /// @brief Weights of the parametric derivatives of each velocity
    ///        component at the closest points, with size
    ///        (num_points, simplex_size(m)), as
    ///        BoundaryProjection::shift_weights holds them
    Eigen::MatrixX<T> weights;

    /// @brief Weights of the parametric derivatives of the pressure at the
    ///        closest points, one matrix per velocity component of the size
    ///        of weights, as BoundaryProjection::coupled_weights holds them
    std::vector<Eigen::MatrixX<T>> pressure_weights;

    /// @brief Weights of the Cartesian derivatives of each component of f
    ///        at the closest points, with size
    ///        (num_points, simplex_size(m − 2)), and no columns below m = 2
    Eigen::MatrixX<T> source_weights;
};

/**
 * @brief Shift of the velocity of a Dirichlet condition rewritten with the
 *        Stokes equation, from the series of the velocity at the closest
 *        points along the lines to the points
 *
 *     −μ Δu + ∇p = f,    −Δu_c = (f_c − ∂_c p) / μ
 *
 * @details Each component is shifted as poisson_shift() shifts a field
 *          whose source is (f_c − ∂_c p) / μ, so that the momentum equation
 *          gives every derivative along the line from the second on. The
 *          terms of the pressure gradient join the shifted values, and
 *          those of f the data. The equation holds for the solution of
 *          either form of the viscous term, whose velocity is free of
 *          divergence
 *
 * @tparam Basis Basis type of the patch
 *
 * @param patch Patch whose parameter space holds the points
 * @param projection Closest points of the points on a boundary
 * @param order Highest order m of the series
 * @param viscosity Dynamic viscosity μ of the fluid
 * @return The weights of the shift at each point of @p projection
 *
 * @throws std::invalid_argument If @p order is negative, if the map of
 *         @p patch is not affine, or if @p viscosity is not positive
 *
 * @pre @p projection was computed through @p patch
 */
template<typename Basis> requires (Basis::dimension == 2)
StokesShift<typename Basis::Scalar> stokes_shift(
    const Patch<Basis, 2>& patch,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order,
    typename Basis::Scalar viscosity);

} // namespace iguana

#endif // IGUANA_ELEMENT_STOKES_STOKES_SHIFT_HPP
