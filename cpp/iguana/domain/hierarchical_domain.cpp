/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "hierarchical_domain.hpp"

#include <numeric>
#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
HierarchicalDomain<T, d>::HierarchicalDomain(TensorDomain<T, d> coarse)
    : active_(1),
      offsets_{0, coarse.num_elements()}
{
    active_[0].resize(coarse.num_elements());
    std::iota(active_[0].begin(), active_[0].end(), 0);
    levels_.push_back(std::move(coarse));
}

template class HierarchicalDomain<double, 1>;
template class HierarchicalDomain<double, 2>;
template class HierarchicalDomain<double, 3>;

} // namespace iguana
