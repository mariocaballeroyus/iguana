/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_nurbs.hpp"

#include <stdexcept>
#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorNURBS<T, d>::TensorNURBS(TensorBSpline<T, d> bspline,
                               Eigen::VectorX<T> weights)
    : bspline_(std::move(bspline)),
      weights_(std::move(weights))
{
    if (weights_.size() != bspline_.num_functions())
        throw std::invalid_argument("TensorNURBS: "
                                    "there must be one weight per basis "
                                    "function");

    if (!(weights_.array() > T{0}).all())
        throw std::invalid_argument("TensorNURBS: "
                                    "the weights must be positive");
}

template<std::floating_point T, std::size_t d>
void TensorNURBS<T, d>::eval_on_element(
    const std::array<int, d>& first_active,
    const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values) const
{
    bspline_.eval_on_element(first_active, points, values);

    Eigen::VectorXi actives;
    bspline_.active_on_element(first_active, actives);
    const Eigen::VectorX<T> active_weights = weights_(actives);

    // Weight each B-spline, then divide by the weight function at each point
    values.array().colwise() *= active_weights.array();
    const Eigen::RowVectorX<T> weight_function = values.colwise().sum();
    values.array().rowwise() /= weight_function.array();
}

template class TensorNURBS<double, 1>;
template class TensorNURBS<double, 2>;
template class TensorNURBS<double, 3>;

} // namespace iguana
