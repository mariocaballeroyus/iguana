/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "hierarchical_grid_iterator.hpp"

#include "iguana/grid/hierarchical_grid.hpp"
#include "iguana/grid/knot_vector.hpp"
#include "iguana/grid/tensor_grid.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
HierarchicalGridIterator<T, d>::HierarchicalGridIterator(
    const HierarchicalGrid<T, d>& grid) noexcept
    : grid_(&grid),
      num_elements_(grid.num_elements())
{
    settle();
}

template<std::floating_point T, std::size_t d>
HierarchicalGridIterator<T, d>&
HierarchicalGridIterator<T, d>::operator++() noexcept
{
    ++index_;
    ++position_;
    settle();

    return *this;
}

template<std::floating_point T, std::size_t d>
void HierarchicalGridIterator<T, d>::settle() noexcept
{
    if (index_ >= num_elements_)
        return;

    // An element remains, so a level with active elements left follows.
    // Levels may have none, once all their elements are refined
    while (position_ ==
           static_cast<int>(grid_->active_elements(level_).size())) {
        ++level_;
        position_ = 0;
    }

    update();
}

template<std::floating_point T, std::size_t d>
void HierarchicalGridIterator<T, d>::update() noexcept
{
    const TensorGrid<T, d>& level_grid = grid_->level(level_);

    std::array<int, d> element_counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        element_counts[direction] =
            level_grid.knots(direction).num_elements();

    const std::array<int, d> axis_elements = unflatten(
        grid_->active_elements(level_)[position_], element_counts);

    for (std::size_t direction = 0; direction < d; ++direction) {
        const KnotVector<T>& knots = level_grid.knots(direction);
        const int element = axis_elements[direction];

        // The functions active on a span start the degree before it
        first_active_[direction] =
            knots.element_span(element) - knots.degree();
        start_[direction] = knots.element_start(element);
        end_[direction] = knots.element_end(element);
    }
}

template class HierarchicalGridIterator<double, 1>;
template class HierarchicalGridIterator<double, 2>;
template class HierarchicalGridIterator<double, 3>;

} // namespace iguana
