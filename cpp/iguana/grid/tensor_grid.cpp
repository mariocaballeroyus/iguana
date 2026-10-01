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
TensorGridIterator<T, d> TensorGrid<T, d>::begin() const noexcept
{
    return TensorGridIterator<T, d>(*this);
}

template class TensorGrid<double, 1>;
template class TensorGrid<double, 2>;
template class TensorGrid<double, 3>;

} // namespace iguana
