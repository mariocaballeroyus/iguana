/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "ghost_faces.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include "iguana/grid/tensor_grid_iterator.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
GhostFaces<T, d>::GhostFaces(const TensorGrid<T, d>& grid,
                             const CellClassification<T, d>& classification)
{
    if (classification.num_elements() != grid.num_elements())
        throw std::invalid_argument("GhostFaces: "
                                    "the classification must have one cell "
                                    "type per element");

    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] = grid.knots(direction).num_elements();

    // Each interior face is found once, from the cell before it
    for (const TensorGridIterator<T, d>& element : grid) {
        const std::array<int, d> cell = unflatten(element.index(), counts);
        const CellType before = classification.cell_type(element.index());

        for (std::size_t direction = 0; direction < d; ++direction) {
            std::array<int, d> next = cell;
            next[direction] += 1;

            if (next[direction] == counts[direction])
                continue;

            const int neighbor = flatten(next, counts);
            const CellType after = classification.cell_type(neighbor);

            const bool cut = before == CellType::cut
                             || after == CellType::cut;
            const bool outside = before == CellType::outside
                                 && after == CellType::outside;

            if (!cut && !outside)
                continue;

            // The face is the cell flattened onto its end in this direction
            Face face{static_cast<int>(direction), element.index(), neighbor,
                      element.start(), element.end()};
            face.start[direction] = element.end()[direction];

            faces_.push_back(face);
        }
    }
}

template class GhostFaces<double, 2>;
template class GhostFaces<double, 3>;

} // namespace iguana
