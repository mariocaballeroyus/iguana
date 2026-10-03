/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "patch.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/LU>

namespace iguana
{

using Eigen::placeholders::all;

namespace
{

/// @brief Jacobian of a patch at a point, with one column per parametric
///        direction
template<typename Scalar, std::size_t n, std::size_t d>
Eigen::Matrix<Scalar, n, d> jacobian_at(
    const std::array<PointMatrix<Scalar, n>, d>& tangents,
    Eigen::Index point)
{
    Eigen::Matrix<Scalar, n, d> jacobian;

    for (std::size_t direction = 0; direction < d; ++direction)
        jacobian.col(direction) = tangents[direction].row(point).transpose();

    return jacobian;
}

} // namespace

template<typename Basis, std::size_t n>
Patch<Basis, n>::Patch(Basis basis, PointMatrix<Scalar, n> coefficients)
    : basis_(std::move(basis)),
      coefficients_(std::move(coefficients))
{
    if (coefficients_.rows() != basis_.num_functions())
        throw std::invalid_argument("Patch: "
                                    "there must be one control point per "
                                    "basis function");
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::position_on_element(
    const Eigen::VectorXi& actives, const Eigen::MatrixX<Scalar>& values,
    PointMatrix<Scalar, n>& positions) const
{
    // Reuse the output buffer when its shape is unchanged
    positions.resize(values.cols(), n);

    // Gather the control points of the active functions, then weight each
    // of them by the value of its function at every point
    positions.noalias() = values.transpose() * coefficients_(actives, all);
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::tangent_on_element(
    const Eigen::VectorXi& actives,
    const std::array<Eigen::MatrixX<Scalar>, dim>& gradients,
    std::array<PointMatrix<Scalar, n>, dim>& tangents) const
{
    for (std::size_t direction = 0; direction < dim; ++direction) {
        // Weight the control points of the active functions by the
        // derivatives of their functions along the direction
        tangents[direction].noalias() =
            gradients[direction].transpose() * coefficients_(actives, all);
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::measure_on_element(
    const std::array<PointMatrix<Scalar, n>, dim>& tangents,
    Eigen::VectorX<Scalar>& measures)
{
    measures.resize(tangents[0].rows());

    for (Eigen::Index point = 0; point < measures.size(); ++point) {
        const Eigen::Matrix<Scalar, n, dim> jacobian =
            jacobian_at<Scalar, n, dim>(tangents, point);

        if constexpr (dim == n)
            measures(point) = std::abs(jacobian.determinant());
        else if constexpr (dim < n)
            measures(point) =
                std::sqrt((jacobian.transpose() * jacobian).determinant());
    }
}

// Curves in the plane and in space, planar regions, surfaces in space and
// volumes, on B-splines and on NURBS
template class Patch<TensorBSpline<double, 1>, 2>;
template class Patch<TensorBSpline<double, 1>, 3>;
template class Patch<TensorBSpline<double, 2>, 2>;
template class Patch<TensorBSpline<double, 2>, 3>;
template class Patch<TensorBSpline<double, 3>, 3>;
template class Patch<TensorNURBS<double, 1>, 2>;
template class Patch<TensorNURBS<double, 1>, 3>;
template class Patch<TensorNURBS<double, 2>, 2>;
template class Patch<TensorNURBS<double, 2>, 3>;
template class Patch<TensorNURBS<double, 3>, 3>;

} // namespace iguana
