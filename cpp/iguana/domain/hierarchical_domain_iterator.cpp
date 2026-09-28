/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "hierarchical_domain_iterator.hpp"

#include "iguana/domain/hierarchical_domain.hpp"
#include "iguana/domain/knot_vector.hpp"
#include "iguana/domain/tensor_domain.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
HierarchicalDomainIterator<T, d>::HierarchicalDomainIterator(
    const HierarchicalDomain<T, d>& domain) noexcept
    : domain_(&domain),
      num_elements_(domain.num_elements())
{
    settle();
}

template<std::floating_point T, std::size_t d>
HierarchicalDomainIterator<T, d>&
HierarchicalDomainIterator<T, d>::operator++() noexcept
{
    ++index_;
    ++position_;
    settle();

    return *this;
}

template<std::floating_point T, std::size_t d>
void HierarchicalDomainIterator<T, d>::settle() noexcept
{
    if (index_ >= num_elements_)
        return;

    // An element remains, so a level with active elements left follows.
    // Levels may have none, once all their elements are refined
    while (position_ ==
           static_cast<int>(domain_->active_elements(level_).size())) {
        ++level_;
        position_ = 0;
    }

    update();
}

template<std::floating_point T, std::size_t d>
void HierarchicalDomainIterator<T, d>::update() noexcept
{
    const TensorDomain<T, d>& level_domain = domain_->level(level_);

    std::array<int, d> element_counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        element_counts[direction] =
            level_domain.knots(direction).num_elements();

    const std::array<int, d> axis_elements = unflatten(
        domain_->active_elements(level_)[position_], element_counts);

    for (std::size_t direction = 0; direction < d; ++direction) {
        const KnotVector<T>& knots = level_domain.knots(direction);
        const int element = axis_elements[direction];

        // The functions active on a span start the degree before it
        first_active_[direction] =
            knots.element_span(element) - knots.degree();
        start_[direction] = knots.element_start(element);
        end_[direction] = knots.element_end(element);
    }
}

template class HierarchicalDomainIterator<double, 1>;
template class HierarchicalDomainIterator<double, 2>;
template class HierarchicalDomainIterator<double, 3>;

} // namespace iguana
