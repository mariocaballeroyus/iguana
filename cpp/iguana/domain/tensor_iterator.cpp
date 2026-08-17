/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "tensor_iterator.hpp"

#include "iguana/multi_index.hpp"

namespace iguana
{

template<int d, std::floating_point T>
TensorDomainIterator<d, T>::TensorDomainIterator(
    const TensorProductBSpline<d, T>& basis) noexcept
    : element_{}, basis_(&basis), each_{}, counts_{}
{
    for (int k = 0; k < d; ++k)
        counts_[k] = basis_->axis(k).num_elements();

    update();
}

template<int d, std::floating_point T>
void TensorDomainIterator<d, T>::update() noexcept
{
    for (int k = 0; k < d; ++k) {
        const BSpline<T>& axis = basis_->axis(k);

        element_.first_active[k] = axis.first_active(each_[k]);
        element_.start[k] = axis.element_start(each_[k]);
        element_.end[k] = axis.element_end(each_[k]);
    }
}

template<int d, std::floating_point T>
TensorDomainIterator<d, T>&
TensorDomainIterator<d, T>::operator++() noexcept
{
    ++element_.index;

    // The element of every direction is advanced rather than split
    // out of the index anew, so that the sweep divides nothing
    if (next_lexicographic(each_, counts_))
        update();

    return *this;
}

template<int d, std::floating_point T>
bool TensorDomainIterator<d, T>::operator==(
    std::default_sentinel_t) const noexcept
{
    return element_.index >= basis_->num_elements();
}

template class TensorDomainIterator<1, double>;
template class TensorDomainIterator<2, double>;
template class TensorDomainIterator<3, double>;

} // namespace iguana
