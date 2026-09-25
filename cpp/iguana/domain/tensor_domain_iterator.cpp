/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_domain_iterator.hpp"

#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorDomainIterator<T, d>::TensorDomainIterator(
    const TensorBSpline<T, d>& basis) noexcept
    : basis_(&basis),
      num_elements_(basis.num_elements())
{
    for (std::size_t direction = 0; direction < d; ++direction)
        element_counts_[direction] = basis.axis(direction).num_elements();

    update();
}

template<std::floating_point T, std::size_t d>
TensorDomainIterator<T, d>& TensorDomainIterator<T, d>::operator++() noexcept
{
    ++index_;

    // The element of each direction advances, the first one fastest
    if (next_lexicographic(axis_elements_, element_counts_))
        update();

    return *this;
}

template<std::floating_point T, std::size_t d>
void TensorDomainIterator<T, d>::update() noexcept
{
    for (std::size_t direction = 0; direction < d; ++direction) {
        const BSpline<T>& axis = basis_->axis(direction);
        const int element = axis_elements_[direction];

        first_active_[direction] = axis.first_active(element);
        start_[direction] = axis.element_start(element);
        end_[direction] = axis.element_end(element);
    }
}

template class TensorDomainIterator<double, 1>;
template class TensorDomainIterator<double, 2>;
template class TensorDomainIterator<double, 3>;

} // namespace iguana
