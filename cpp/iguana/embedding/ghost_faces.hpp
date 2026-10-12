/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_GHOST_FACES_HPP
#define IGUANA_EMBEDDING_GHOST_FACES_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "iguana/embedding/cell_classification.hpp"
#include "iguana/grid/tensor_grid.hpp"

namespace iguana
{

/**
 * @brief Interior faces of a grid along which a ghost penalty extends the
 *        solution beyond the physical domain
 *
 * With alpha the volume fraction of a cell, the penalty acts where the two
 * cells of a face are not both full, {{alpha}} < 1, nor one full and the
 * other empty, |[[alpha]]| < 1. In cell types, these are the faces of the
 * cut cells and those between two outside cells, so that the penalty
 * reaches every function that the physical domain alone leaves undetermined
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
std::vector<int> ghost_faces(const TensorGrid<T, d>& grid,
                             const CellClassification<T, d>& classification);

} // namespace iguana

#endif // IGUANA_EMBEDDING_GHOST_FACES_HPP
