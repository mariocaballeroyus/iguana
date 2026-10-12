/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_JUMP_FACES_HPP
#define IGUANA_EMBEDDING_JUMP_FACES_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "iguana/embedding/cell_classification.hpp"
#include "iguana/grid/tensor_grid.hpp"

namespace iguana
{

/**
 * @brief Interior faces of a grid across which the volume fraction of the
 *        cells may jump
 *
 * With alpha the volume fraction of a cell, the generalized shifted
 * boundary method imposes the conditions of the physical boundary on the
 * faces where [[alpha]] is not zero. In cell types, that can only happen
 * between cells of different types or between two cut cells, whose
 * fractions the assembly compares, and never between two inside or two
 * outside cells. With every cell inside or outside, these are the faces of
 * the surrogate boundary between two cells
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 *
 * @param grid Grid whose elements are the cells
 * @param classification Cell type of each element of the grid
 * @return Indices of the faces in the numbering of the grid, increasing
 *
 * @throws std::invalid_argument If the classification does not have one
 *         cell type per element of the grid
 */
template<std::floating_point T, std::size_t d>
std::vector<int> jump_faces(const TensorGrid<T, d>& grid,
                            const CellClassification<T, d>& classification);

} // namespace iguana

#endif // IGUANA_EMBEDDING_JUMP_FACES_HPP
