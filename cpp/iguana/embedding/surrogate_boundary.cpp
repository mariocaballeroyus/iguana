/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "surrogate_boundary.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include "iguana/grid/tensor_grid_iterator.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
SurrogateBoundary<T, d>::SurrogateBoundary(const TensorGrid<T, d>& grid,
                                           const EmbeddedDomain<T, d>& domain)
{
    if (domain.num_elements() != grid.num_elements())
        throw std::invalid_argument("SurrogateBoundary: "
                                    "the domain must have one cell type per "
                                    "element");

    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] = grid.knots(direction).num_elements();

    // Whether the neighbor of a cell across one of its faces is inside, a
    // neighbor past the edge of the grid being outside
    const auto is_inside = [&](std::array<int, d> cell, std::size_t direction,
                               int side) {
        cell[direction] += side;

        if (cell[direction] < 0 || cell[direction] >= counts[direction])
            return false;

        return domain.cell_type(flatten(cell, counts)) == CellType::inside;
    };

    offsets_.resize(grid.num_elements() + 1);
    offsets_(0) = 0;

    for (const TensorGridIterator<T, d>& element : grid) {
        if (domain.cell_type(element.index()) == CellType::inside) {
            const std::array<int, d> cell = unflatten(element.index(), counts);

            for (std::size_t direction = 0; direction < d; ++direction) {
                for (const int side : {-1, 1}) {
                    if (is_inside(cell, direction, side))
                        continue;

                    // The face is the cell flattened onto its start or end
                    // in this direction
                    Face face{static_cast<int>(direction), side,
                              element.start(), element.end()};
                    const T level = side < 0 ? element.start()[direction]
                                             : element.end()[direction];
                    face.start[direction] = level;
                    face.end[direction] = level;

                    faces_.push_back(face);
                }
            }
        }

        offsets_(element.index() + 1) = num_faces();
    }
}

template class SurrogateBoundary<double, 2>;
template class SurrogateBoundary<double, 3>;

} // namespace iguana
