/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "directional_derivs.hpp"

#include <stdexcept>

#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T>
Eigen::MatrixX<T> directional_derivs(const Eigen::Matrix2<T>& frame,
                                     int order)
{
    if (order < 0)
        throw std::invalid_argument("directional_derivs: "
                                    "the order must be non-negative");

    const int size = simplex_size(order);
    Eigen::MatrixX<T> derivs = Eigen::MatrixX<T>::Zero(size, size);
    derivs(0, 0) = T{1};

    // Each derivative is the one an order below after one more along f₀,
    // or along f₁ for those with none along f₀, which raises each partial
    // derivative by one along each coordinate, times that component of f
    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j + i <= order; ++j) {
            if (j + i == 0)
                continue;

            const int row = simplex_index(j, i, order);
            const int below = j > 0 ? simplex_index(j - 1, i, order)
                                    : simplex_index(0, i - 1, order);
            const Eigen::Vector2<T> direction = frame.col(j > 0 ? 0 : 1);

            for (int b = 0; b < j + i; ++b) {
                for (int a = 0; a + b < j + i; ++a) {
                    const T weight = derivs(below, simplex_index(a, b, order));

                    derivs(row, simplex_index(a + 1, b, order)) +=
                        direction(0) * weight;
                    derivs(row, simplex_index(a, b + 1, order)) +=
                        direction(1) * weight;
                }
            }
        }
    }

    return derivs;
}

template Eigen::MatrixX<double> directional_derivs(
    const Eigen::Matrix2<double>&, int);

} // namespace iguana
