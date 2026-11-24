/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "surrogate_boundary.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "iguana/grid/hierarchical_grid.hpp"
#include "iguana/grid/tensor_grid.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
template<typename Grid>
    requires (Grid::dimension == d)
SurrogateBoundary<T, d>::SurrogateBoundary(
    const Grid& grid, const CellClassification<T, d>& classification)
{
    const int num_elements = grid.num_elements();

    if (classification.num_elements() != num_elements)
        throw std::invalid_argument("SurrogateBoundary: "
                                    "the classification must have one cell "
                                    "type per element");

    const auto inside = [&](int cell) {
        return classification.cell_type(cell) == CellType::inside;
    };

    // The faces of each inside cell, before they are placed cell by cell
    std::vector<std::vector<Face>> held(
        static_cast<std::size_t>(num_elements));

    // Interior faces between an inside cell and one that is not
    for (int face = 0; face < grid.num_faces(); ++face) {
        const std::array<int, 2> cells = grid.face_elements(face);

        if (inside(cells[0]) == inside(cells[1]))
            continue;

        const int holder = inside(cells[0]) ? cells[0] : cells[1];
        const int side = inside(cells[0]) ? 1 : -1;

        held[static_cast<std::size_t>(holder)].push_back(
            {static_cast<int>(grid.face_direction(face)), side,
             grid.face_start(face), grid.face_end(face)});
    }

    // The sides of inside cells on the edge of the grid, the whole side
    // each, past which every cell counts as outside
    std::array<T, d> lower = grid.element_start(0);
    std::array<T, d> upper = grid.element_end(0);

    for (int cell = 1; cell < num_elements; ++cell) {
        const std::array<T, d> start = grid.element_start(cell);
        const std::array<T, d> end = grid.element_end(cell);

        for (std::size_t direction = 0; direction < d; ++direction) {
            lower[direction] = std::min(lower[direction], start[direction]);
            upper[direction] = std::max(upper[direction], end[direction]);
        }
    }

    for (int cell = 0; cell < num_elements; ++cell) {
        if (!inside(cell))
            continue;

        const std::array<T, d> start = grid.element_start(cell);
        const std::array<T, d> end = grid.element_end(cell);

        for (std::size_t direction = 0; direction < d; ++direction) {
            for (const int side : {-1, 1}) {
                const T level = side < 0 ? start[direction] : end[direction];

                if (level != (side < 0 ? lower : upper)[direction])
                    continue;

                // The cell flattened onto its start or end in this direction
                Face face{static_cast<int>(direction), side, start, end};
                face.start[direction] = level;
                face.end[direction] = level;

                held[static_cast<std::size_t>(cell)].push_back(face);
            }
        }
    }

    // Each cell's faces by direction, the start side before the end one,
    // and then along the side
    offsets_.resize(num_elements + 1);
    offsets_(0) = 0;

    for (int cell = 0; cell < num_elements; ++cell) {
        std::vector<Face>& faces = held[static_cast<std::size_t>(cell)];

        std::ranges::sort(faces, {}, [](const Face& face) {
            return std::tuple(face.direction, face.side, face.start);
        });
        faces_.insert(faces_.end(), faces.begin(), faces.end());
        offsets_(cell + 1) = num_faces();
    }
}

template class SurrogateBoundary<double, 2>;
template class SurrogateBoundary<double, 3>;

template SurrogateBoundary<double, 2>::SurrogateBoundary(
    const TensorGrid<double, 2>&, const CellClassification<double, 2>&);
template SurrogateBoundary<double, 3>::SurrogateBoundary(
    const TensorGrid<double, 3>&, const CellClassification<double, 3>&);
template SurrogateBoundary<double, 2>::SurrogateBoundary(
    const HierarchicalGrid<double, 2>&, const CellClassification<double, 2>&);
template SurrogateBoundary<double, 3>::SurrogateBoundary(
    const HierarchicalGrid<double, 3>&, const CellClassification<double, 3>&);

} // namespace iguana
