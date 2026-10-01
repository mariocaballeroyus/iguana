/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_grid_iterator.hpp"

#include "iguana/grid/knot_vector.hpp"
#include "iguana/grid/tensor_grid.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorGridIterator<T, d>::TensorGridIterator(
    const TensorGrid<T, d>& grid) noexcept
    : grid_(&grid),
      num_elements_(grid.num_elements())
{
    for (std::size_t direction = 0; direction < d; ++direction)
        element_counts_[direction] = grid.knots(direction).num_elements();

    update();
}

template<std::floating_point T, std::size_t d>
TensorGridIterator<T, d>& TensorGridIterator<T, d>::operator++() noexcept
{
    ++index_;

    // The element of each direction advances, the first one fastest
    if (next_lexicographic(axis_elements_, element_counts_))
        update();

    return *this;
}

template<std::floating_point T, std::size_t d>
void TensorGridIterator<T, d>::update() noexcept
{
    for (std::size_t direction = 0; direction < d; ++direction) {
        const KnotVector<T>& knots = grid_->knots(direction);
        const int element = axis_elements_[direction];

        // The functions active on a span start the degree before it
        first_active_[direction] =
            knots.element_span(element) - knots.degree();
        start_[direction] = knots.element_start(element);
        end_[direction] = knots.element_end(element);
    }
}

template class TensorGridIterator<double, 1>;
template class TensorGridIterator<double, 2>;
template class TensorGridIterator<double, 3>;

} // namespace iguana
