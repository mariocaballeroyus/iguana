/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_grid.hpp"

#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorGrid<T, d>::TensorGrid(std::array<KnotVector<T>, d> knots)
    : knots_(std::move(knots)),
      num_elements_(1)
{
    for (const KnotVector<T>& axis_knots : knots_)
        num_elements_ *= axis_knots.num_elements();
}

template<std::floating_point T, std::size_t d>
std::array<std::vector<T>, d> TensorGrid<T, d>::lines() const
{
    std::array<std::vector<T>, d> result;

    for (std::size_t direction = 0; direction < d; ++direction) {
        const KnotVector<T>& axis_knots = knots_[direction];

        for (int element = 0; element < axis_knots.num_elements(); ++element)
            result[direction].push_back(axis_knots.element_start(element));

        result[direction].push_back(axis_knots.domain_end());
    }

    return result;
}

template<std::floating_point T, std::size_t d>
TensorGridIterator<T, d> TensorGrid<T, d>::begin() const noexcept
{
    return TensorGridIterator<T, d>(*this);
}

template class TensorGrid<double, 1>;
template class TensorGrid<double, 2>;
template class TensorGrid<double, 3>;

} // namespace iguana
