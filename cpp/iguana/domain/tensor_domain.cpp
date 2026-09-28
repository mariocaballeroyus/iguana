/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_domain.hpp"

#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorDomain<T, d>::TensorDomain(std::array<KnotVector<T>, d> knots)
    : knots_(std::move(knots)),
      num_elements_(1)
{
    for (const KnotVector<T>& axis_knots : knots_)
        num_elements_ *= axis_knots.num_elements();
}

template class TensorDomain<double, 1>;
template class TensorDomain<double, 2>;
template class TensorDomain<double, 3>;

} // namespace iguana
