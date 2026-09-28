/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "embedding.hpp"

#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
Embedding<T, d>::Embedding(std::vector<CellType> cell_types)
    : cell_types_(std::move(cell_types))
{
}

template class Embedding<double, 1>;
template class Embedding<double, 2>;
template class Embedding<double, 3>;

} // namespace iguana
