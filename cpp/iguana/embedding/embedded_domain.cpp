/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "embedded_domain.hpp"

#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
EmbeddedDomain<T, d>::EmbeddedDomain(std::vector<CellType> cell_types)
    : cell_types_(std::move(cell_types))
{
}

template class EmbeddedDomain<double, 1>;
template class EmbeddedDomain<double, 2>;
template class EmbeddedDomain<double, 3>;

} // namespace iguana
