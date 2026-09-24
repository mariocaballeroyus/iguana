/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_domain.hpp"

#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorDomain<T, d>::TensorDomain(Patch<T, d> patch)
    : patch_(std::move(patch))
{
}

template<std::floating_point T, std::size_t d>
TensorDomainIterator<T, d> TensorDomain<T, d>::begin() const noexcept
{
    return TensorDomainIterator<T, d>(patch_.basis());
}

template class TensorDomain<double, 1>;
template class TensorDomain<double, 2>;
template class TensorDomain<double, 3>;

} // namespace iguana
