/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_grid.hpp"

#include <utility>

#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorGrid<T, d>::TensorGrid(std::array<KnotVector<T>, d> knots)
    : knots_(std::move(knots)),
      num_elements_(1)
{
    for (const KnotVector<T>& axis_knots : knots_)
        num_elements_ *= axis_knots.num_elements();

    for (std::size_t direction = 0; direction < d; ++direction) {
        const int count = knots_[direction].num_elements();

        face_offsets_[direction + 1] = face_offsets_[direction]
                                       + num_elements_ / count * (count - 1);
    }
}

template<std::floating_point T, std::size_t d>
std::size_t TensorGrid<T, d>::face_direction(int face) const noexcept
{
    std::size_t direction = 0;

    while (face >= face_offsets_[direction + 1])
        ++direction;

    return direction;
}

template<std::floating_point T, std::size_t d>
std::array<int, 2> TensorGrid<T, d>::face_elements(int face) const noexcept
{
    const std::array<int, d> counts = element_counts();
    std::array<int, d> element = element_before(face);
    const int before = flatten(element, counts);

    element[face_direction(face)] += 1;

    return {before, flatten(element, counts)};
}

template<std::floating_point T, std::size_t d>
std::array<T, d> TensorGrid<T, d>::face_start(int face) const noexcept
{
    const std::size_t direction = face_direction(face);
    const std::array<int, d> element = element_before(face);
    std::array<T, d> start{};

    for (std::size_t axis = 0; axis < d; ++axis)
        start[axis] = knots_[axis].element_start(element[axis]);

    start[direction] = knots_[direction].element_end(element[direction]);

    return start;
}

template<std::floating_point T, std::size_t d>
std::array<T, d> TensorGrid<T, d>::face_end(int face) const noexcept
{
    const std::array<int, d> element = element_before(face);
    std::array<T, d> end{};

    for (std::size_t axis = 0; axis < d; ++axis)
        end[axis] = knots_[axis].element_end(element[axis]);

    return end;
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

template<std::floating_point T, std::size_t d>
std::array<int, d> TensorGrid<T, d>::element_counts() const noexcept
{
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] = knots_[direction].num_elements();

    return counts;
}

template<std::floating_point T, std::size_t d>
std::array<int, d> TensorGrid<T, d>::element_before(int face) const noexcept
{
    const std::size_t direction = face_direction(face);

    std::array<int, d> bounds = element_counts();
    bounds[direction] -= 1;

    return unflatten(face - face_offsets_[direction], bounds);
}

template class TensorGrid<double, 1>;
template class TensorGrid<double, 2>;
template class TensorGrid<double, 3>;

} // namespace iguana
