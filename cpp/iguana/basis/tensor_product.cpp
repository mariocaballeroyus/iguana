/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "tensor_product.hpp"

#include<utility>

#include "iguana/multi_index.hpp"

namespace iguana
{

template<int d, std::floating_point T>
TensorProductBSpline<d, T>::TensorProductBSpline(
    std::array<BSpline<T>, d> axes)
    : axes_(std::move(axes))
{
    num_functions_ = 1;
    num_elements_ = 1;
    num_active_ = 1;

    for (const BSpline<T>& basis : axes_) {
        num_functions_ *= basis.num_functions();
        num_elements_ *= basis.num_elements();
        num_active_ *= basis.num_active();
    }
}

template<int d, std::floating_point T>
std::array<int, d> TensorProductBSpline<d, T>::first_active(
    int element) const noexcept
{
    std::array<int, d> first{};

    // Elements numbered with the first direction running fastest
    int remaining = element;

    for (std::size_t k = 0; k < d; ++k) {
        // Take the remainder of the division by its own count
        const int count = axes_[k].num_elements();
        first[k] = axes_[k].first_active(remaining % count);

        // Pass the quotient on to the next one
        remaining /= count;
    }

    return first;
}

template<int d, std::floating_point T>
void TensorProductBSpline<d, T>::active_on_element(
    int element, Eigen::VectorXi& actives) const
{
    // Actives buffer, reused across elements
    actives.resize(num_active_);

    // Lowest non-zero function of each direction
    const std::array<int, d> first = first_active(element);

    // Bounds of the multi-index, one non-zero function per direction
    std::array<int, d> bounds{};

    for (std::size_t k = 0; k < d; ++k)
        bounds[k] = axes_[k].num_active();

    // Offsets from the lowest function, advanced by next_lexicographic
    std::array<int, d> offset{};
    int position = 0;

    do {
        // Flatten the multi-index
        int index = 0;

        for (std::size_t k = d; k-- > 0;)
            index = index * axes_[k].num_functions() + first[k] + offset[k];

        // The rows of the evaluation follow the same order
        actives[position++] = index;
    } while (next_lexicographic(offset, bounds));
}

template class TensorProductBSpline<1, double>;
template class TensorProductBSpline<2, double>;
template class TensorProductBSpline<3, double>;

} // namespace iguana
