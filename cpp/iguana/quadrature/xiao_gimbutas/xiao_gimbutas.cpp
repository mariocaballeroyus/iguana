/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "xiao_gimbutas.hpp"

#include <array>
#include <vector>

namespace iguana
{

template<std::floating_point T>
XiaoGimbutas<T>::XiaoGimbutas(int degree)
{
    const auto [coordinates, weights] = tabulated_rule(degree);
    const Eigen::Index num_points = weights.size();

    // The coordinates hold one point after another, as the rows
    using RowMajor = Eigen::Matrix<T, Eigen::Dynamic, 2, Eigen::RowMajor>;

    points_ = Eigen::Map<const RowMajor>(coordinates.data(), num_points, 2);
    weights_ = Eigen::Map<const Eigen::VectorX<T>>(weights.data(), num_points);
}

template class XiaoGimbutas<double>;

} // namespace iguana
