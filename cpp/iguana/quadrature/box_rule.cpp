/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "box_rule.hpp"

#include <array>
#include <cstddef>

namespace iguana
{

template<std::floating_point T, std::size_t d>
void BoxRule<T, d>::map_to_parameter_space(const std::array<T, d>& start,
                                           const std::array<T, d>& end,
                                           Eigen::MatrixX<T>& points,
                                           Eigen::VectorX<T>& weights)
{
    T jacobian{1};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const T half = (end[direction] - start[direction]) / T{2};
        const T middle = (start[direction] + end[direction]) / T{2};

        points.col(direction).array() =
            half * points.col(direction).array() + middle;
        jacobian *= half;
    }

    weights *= jacobian;
}

template class BoxRule<double, 1>;
template class BoxRule<double, 2>;
template class BoxRule<double, 3>;

} // namespace iguana
