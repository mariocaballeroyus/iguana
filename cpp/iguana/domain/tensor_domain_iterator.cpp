/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_domain_iterator.hpp"

#include "iguana/domain/knot_vector.hpp"
#include "iguana/domain/tensor_domain.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorDomainIterator<T, d>::TensorDomainIterator(
    const TensorDomain<T, d>& domain) noexcept
    : domain_(&domain),
      num_elements_(domain.num_elements())
{
    for (std::size_t direction = 0; direction < d; ++direction)
        element_counts_[direction] = domain.knots(direction).num_elements();

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
        const KnotVector<T>& knots = domain_->knots(direction);
        const int element = axis_elements_[direction];

        // The functions active on a span start the degree before it
        first_active_[direction] =
            knots.element_span(element) - knots.degree();
        start_[direction] = knots.element_start(element);
        end_[direction] = knots.element_end(element);
    }
}

template class TensorDomainIterator<double, 1>;
template class TensorDomainIterator<double, 2>;
template class TensorDomainIterator<double, 3>;

} // namespace iguana
