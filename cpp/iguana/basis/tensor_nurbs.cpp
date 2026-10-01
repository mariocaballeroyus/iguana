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

namespace
{

/**
 * @brief Turns the values of the active B-splines into those of the rational
 *        functions, R = w N / W, in place
 *
 * @param active_weights Weights of the active functions
 * @param values B-spline values of size (num_active, num_points), which
 *        become the rational values
 *
 * @return Weight function W at each point
 */
template<std::floating_point T>
Eigen::RowVectorX<T> rational_values(const Eigen::VectorX<T>& active_weights,
                                     Eigen::MatrixX<T>& values)
{
    values.array().colwise() *= active_weights.array();
    const Eigen::RowVectorX<T> weight_function = values.colwise().sum();
    values.array().rowwise() /= weight_function.array();

    return weight_function;
}

/**
 * @brief Turns a derivative of the active B-splines into that of the
 *        rational functions, d_a R = (w d_a N - R d_a W) / W, in place
 *
 * @param active_weights Weights of the active functions
 * @param values Rational values R, as rational_values() leaves them
 * @param weight_function Weight function W at each point
 * @param derivative B-spline derivative along a direction, of the size of
 *        @p values, which becomes the rational one
 *
 * @return Derivative of the weight function along the direction, d_a W
 */
template<std::floating_point T>
Eigen::RowVectorX<T> rational_derivative(
    const Eigen::VectorX<T>& active_weights, const Eigen::MatrixX<T>& values,
    const Eigen::RowVectorX<T>& weight_function, Eigen::MatrixX<T>& derivative)
{
    derivative.array().colwise() *= active_weights.array();
    const Eigen::RowVectorX<T> weight_derivative = derivative.colwise().sum();
    derivative.array() -=
        values.array().rowwise() * weight_derivative.array();
    derivative.array().rowwise() /= weight_function.array();

    return weight_derivative;
}

} // namespace

template<std::floating_point T, std::size_t d>
void TensorNURBS<T, d>::eval_on_element(
    const std::array<int, d>& first_active,
    const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values) const
{
    bspline_.eval_on_element(first_active, points, values);

    Eigen::VectorXi actives;
    bspline_.active_on_element(first_active, actives);
    const Eigen::VectorX<T> active_weights = weights_(actives);

    rational_values(active_weights, values);
}

template<std::floating_point T, std::size_t d>
void TensorNURBS<T, d>::grad_on_element(
    const std::array<int, d>& first_active,
    const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values,
    std::array<Eigen::MatrixX<T>, d>& gradients) const
{
    bspline_.grad_on_element(first_active, points, values, gradients);

    Eigen::VectorXi actives;
    bspline_.active_on_element(first_active, actives);
    const Eigen::VectorX<T> active_weights = weights_(actives);

    const Eigen::RowVectorX<T> weight_function =
        rational_values(active_weights, values);

    for (std::size_t direction = 0; direction < d; ++direction)
        rational_derivative(active_weights, values, weight_function,
                            gradients[direction]);
}

template<std::floating_point T, std::size_t d>
void TensorNURBS<T, d>::hess_on_element(
    const std::array<int, d>& first_active,
    const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values,
    std::array<Eigen::MatrixX<T>, d>& gradients,
    std::array<Eigen::MatrixX<T>, d * (d + 1) / 2>& hessians) const
{
    bspline_.hess_on_element(first_active, points, values, gradients,
                             hessians);

    Eigen::VectorXi actives;
    bspline_.active_on_element(first_active, actives);
    const Eigen::VectorX<T> active_weights = weights_(actives);

    const Eigen::RowVectorX<T> weight_function =
        rational_values(active_weights, values);

    // The derivatives of the weight function enter the second ones
    std::array<Eigen::RowVectorX<T>, d> weight_derivatives;

    for (std::size_t direction = 0; direction < d; ++direction)
        weight_derivatives[direction] = rational_derivative(
            active_weights, values, weight_function, gradients[direction]);

    // d_ab R = (w d_ab N - d_a R d_b W - d_b R d_a W - R d_ab W) / W
    const auto second_derivative = [&](Eigen::MatrixX<T>& hessian,
                                       std::size_t a, std::size_t b) {
        hessian.array().colwise() *= active_weights.array();
        const Eigen::RowVectorX<T> weight_second_derivative =
            hessian.colwise().sum();

        hessian.array() -=
            gradients[a].array().rowwise() * weight_derivatives[b].array() +
            gradients[b].array().rowwise() * weight_derivatives[a].array() +
            values.array().rowwise() * weight_second_derivative.array();
        hessian.array().rowwise() /= weight_function.array();
    };

    for (std::size_t direction = 0; direction < d; ++direction)
        second_derivative(hessians[direction], direction, direction);

    // Mixed pairs follow the pure ones, in lexicographic order
    std::size_t pair = d;

    for (std::size_t first = 0; first < d; ++first) {
        for (std::size_t second = first + 1; second < d; ++second)
            second_derivative(hessians[pair++], first, second);
    }
}

template class TensorNURBS<double, 1>;
template class TensorNURBS<double, 2>;
template class TensorNURBS<double, 3>;

} // namespace iguana
