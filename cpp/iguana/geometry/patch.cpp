/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "patch.hpp"

#include <stdexcept>
#include <utility>

namespace iguana
{

using Eigen::placeholders::all;

template<std::floating_point T, std::size_t d, std::size_t n>
Patch<T, d, n>::Patch(TensorBSpline<T, d> basis,
                      PointMatrix<T, n> coefficients)
    : basis_(std::move(basis)),
      coefficients_(std::move(coefficients))
{
    if (coefficients_.rows() != basis_.num_functions())
        throw std::invalid_argument("Patch: "
                                    "there must be one control point per "
                                    "basis function");
}

template<std::floating_point T, std::size_t d, std::size_t n>
void Patch<T, d, n>::position_on_element(const Eigen::VectorXi& actives,
                                         const Eigen::MatrixX<T>& values,
                                         PointMatrix<T, n>& positions) const
{
    // Reuse the output buffer when its shape is unchanged
    positions.resize(values.cols(), n);

    // Gather the control points of the active functions, then weight each
    // of them by the value of its function at every point
    positions.noalias() = values.transpose() * coefficients_(actives, all);
}

template<std::floating_point T, std::size_t d, std::size_t n>
void Patch<T, d, n>::tangents_on_element(
    const Eigen::VectorXi& actives,
    const std::array<Eigen::MatrixX<T>, d>& gradients,
    std::array<PointMatrix<T, n>, d>& tangents) const
{
    for (std::size_t direction = 0; direction < d; ++direction) {
        // Weight the control points of the active functions by the
        // derivatives of their functions along the direction
        tangents[direction].noalias() =
            gradients[direction].transpose() * coefficients_(actives, all);
    }
}

// Curves in the plane and in space, planar regions, surfaces in space and
// volumes
template class Patch<double, 1, 2>;
template class Patch<double, 1, 3>;
template class Patch<double, 2, 2>;
template class Patch<double, 2, 3>;
template class Patch<double, 3, 3>;

} // namespace iguana
