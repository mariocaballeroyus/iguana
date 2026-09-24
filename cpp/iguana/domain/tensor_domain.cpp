/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_domain.hpp"

#include <stdexcept>
#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorDomain<T, d>::TensorDomain(Patch<T, d> patch)
    : patch_(std::move(patch)),
      cell_types_(static_cast<std::size_t>(patch_.basis().num_elements()),
                  CellType::inside)
{
}

template<std::floating_point T, std::size_t d>
TensorDomain<T, d>::TensorDomain(Patch<T, d> patch,
                                 std::vector<CellType> cell_types)
    : patch_(std::move(patch)),
      cell_types_(std::move(cell_types))
{
    if (cell_types_.size() != static_cast<std::size_t>(num_elements()))
        throw std::invalid_argument("TensorDomain: "
                                    "there must be one cell type per element");
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
