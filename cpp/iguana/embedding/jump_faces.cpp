/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "jump_faces.hpp"

#include <array>
#include <stdexcept>

namespace iguana
{

template<std::floating_point T, std::size_t d>
std::vector<int> jump_faces(const TensorGrid<T, d>& grid,
                            const CellClassification<T, d>& classification)
{
    if (classification.num_elements() != grid.num_elements())
        throw std::invalid_argument("jump_faces: "
                                    "the classification must have one cell "
                                    "type per element");

    std::vector<int> faces;

    for (int face = 0; face < grid.num_faces(); ++face) {
        const std::array<int, 2> cells = grid.face_elements(face);
        const CellType before = classification.cell_type(cells[0]);
        const CellType after = classification.cell_type(cells[1]);

        // Cells of one type are both full or both empty, unless cut
        if (before != after || before == CellType::cut)
            faces.push_back(face);
    }

    return faces;
}

template std::vector<int> jump_faces(const TensorGrid<double, 2>&,
                                     const CellClassification<double, 2>&);
template std::vector<int> jump_faces(const TensorGrid<double, 3>&,
                                     const CellClassification<double, 3>&);

} // namespace iguana
