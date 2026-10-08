/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "cell_classification.hpp"

#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
CellClassification<T, d>::CellClassification(std::vector<CellType> cell_types)
    : cell_types_(std::move(cell_types))
{
}

template class CellClassification<double, 1>;
template class CellClassification<double, 2>;
template class CellClassification<double, 3>;

} // namespace iguana
