/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_bspline.hpp"

#include <utility>

#include "iguana/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorBSpline<T, d>::TensorBSpline(std::array<BSpline<T>, d> axes)
    : axes_(std::move(axes)),
      num_functions_(1),
      num_elements_(1),
      num_active_(1)
{
    for (const BSpline<T>& axis : axes_) {
        num_functions_ *= axis.num_functions();
        num_elements_ *= axis.num_elements();
        num_active_ *= axis.num_active();
    }
}

template<std::floating_point T, std::size_t d>
void TensorBSpline<T, d>::active_on_element(
    int element, Eigen::VectorXi& actives) const
{
    // Decode the flat element index, with the first direction fastest
    std::array<int, d> first_active{};
    std::array<int, d> active_counts{};
    std::array<int, d> function_counts{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const BSpline<T>& axis = axes_[direction];
        const int axis_element = element % axis.num_elements();

        first_active[direction] = axis.first_active(axis_element);
        active_counts[direction] = axis.num_active();
        function_counts[direction] = axis.num_functions();
        element /= axis.num_elements();
    }

    actives.resize(num_active_);

    // Enumerate local active offsets, then flatten their global indices
    std::array<int, d> offset{};
    std::array<int, d> index{};
    int position = 0;

    do {
        for (std::size_t direction = 0; direction < d; ++direction)
            index[direction] = first_active[direction] + offset[direction];

        actives[position++] = flatten(index, function_counts);
    } while (next_lexicographic(offset, active_counts));
}

template class TensorBSpline<double, 1>;
template class TensorBSpline<double, 2>;
template class TensorBSpline<double, 3>;

} // namespace iguana
