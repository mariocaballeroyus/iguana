/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_FIELD_INTERPOLATION_HPP
#define IGUANA_GEOMETRY_FIELD_INTERPOLATION_HPP

#include <concepts>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Collocation matrix of a univariate basis at its Greville
 *        abscissae, whose solve interpolates values given there
 *
 * @tparam T Floating-point type
 *
 * @param bspline Univariate basis
 * @return Matrix with one row per abscissa and one column per function
 */
template<std::floating_point T>
Eigen::MatrixX<T> collocation_matrix(const BSpline<T>& bspline);

/**
 * @brief Points at which a field is interpolated in the basis of a planar
 *        patch, the images of its Greville points, one per function in
 *        their order
 *
 * @tparam T Floating-point type
 *
 * @param patch Patch with an affine map
 * @return Points of size (num_functions, 2), one per row
 *
 * @throws std::invalid_argument If the map of @p patch is not affine
 */
template<std::floating_point T>
PointMatrix<T, 2> interpolation_points(
    const Patch<TensorBSpline<T, 2>, 2>& patch);

/**
 * @brief Cartesian derivatives at points of the field that interpolates
 *        values at the interpolation points of a planar patch in its basis
 *
 *     f_h = Σ_i c_i N_i,    f_h(x_k) = f_k at interpolation_points()
 *
 * @details The interpolant is a spline of the basis, so its derivatives
 *          are the exact ones of a piecewise polynomial, those above the
 *          degree vanishing, and each point takes those of the element
 *          holding it. Columns follow simplex_index(a, b, order) for
 *          ∂_x^a ∂_y^b, with a running fastest
 *
 * @tparam T Floating-point type
 *
 * @param patch Patch with an affine map, whose basis spans the field
 * @param values Values of the field at interpolation_points(), one per
 *        function
 * @param points Physical points, one per row
 * @param order Highest total order of the derivatives
 * @return Derivatives of size (num_points, simplex_size(order))
 *
 * @throws std::invalid_argument If @p order is negative or above the
 *         degree of the basis in a direction, if the map of @p patch is not
 *         affine, or if @p values does not hold one value per function
 *
 * @pre The points lie in the patch
 */
template<std::floating_point T>
Eigen::MatrixX<T> interpolated_derivs(
    const Patch<TensorBSpline<T, 2>, 2>& patch,
    const Eigen::VectorX<T>& values, const PointMatrix<T, 2>& points,
    int order);

} // namespace iguana

#endif // IGUANA_GEOMETRY_FIELD_INTERPOLATION_HPP
