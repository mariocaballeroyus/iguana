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

template<std::floating_point T, std::size_t d>
Patch<T, d>::Patch(TensorBSpline<T, d> basis,
                   Eigen::MatrixX3<T> coefficients)
    : basis_(std::move(basis)),
      coefficients_(std::move(coefficients))
{
    if (coefficients_.rows() != basis_.num_functions())
        throw std::invalid_argument("Patch: "
                                    "there must be one control point per "
                                    "basis function");
}

template<std::floating_point T, std::size_t d>
void Patch<T, d>::position_on_element(const Eigen::VectorXi& actives,
                                      const Eigen::MatrixX<T>& values,
                                      Eigen::MatrixX3<T>& positions) const
{
    // Reuse the output buffer when its shape is unchanged
    positions.resize(values.cols(), 3);

    // Gather the control points of the active functions, then weight each
    // of them by the value of its function at every point
    positions.noalias() = values.transpose() * coefficients_(actives, all);
}

template class Patch<double, 1>;
template class Patch<double, 2>;
template class Patch<double, 3>;

} // namespace iguana
