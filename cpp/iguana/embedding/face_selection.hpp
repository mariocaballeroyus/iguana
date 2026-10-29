/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_FACE_SELECTION_HPP
#define IGUANA_EMBEDDING_FACE_SELECTION_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "iguana/embedding/cell_classification.hpp"
#include "iguana/grid/tensor_grid.hpp"

namespace iguana
{

/**
 * @brief Kind of the faces a ghost penalty acts on
 *
 * The faces of the cut cells and those between two outside cells partition
 * the faces of a ghost penalty, so that two penalties of different
 * strengths may act on each
 */
enum class GhostFaces
{
    /// @brief The faces of the cut cells and those between outside cells
    all,

    /// @brief The faces of the cut cells
    cut,

    /// @brief The faces between two outside cells
    outside
};

/**
 * @brief Interior faces of a grid along which a ghost penalty extends the
 *        solution beyond the physical domain
 *
 * @details Used in the Generalized Shifted Boundary Method to penalize the
 *          derivative jumps on faces with mean volume fraction below one and
 *          volume fraction jump below one in magnitude: the faces of the cut
 *          cells and those between two outside cells, or either kind alone
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 *
 * @param grid Grid whose elements are the cells
 * @param classification Cell type of each element of the grid
 * @param faces Kind of the faces selected
 * @return Indices of the faces in the numbering of the grid, increasing
 *
 * @throws std::invalid_argument If the classification does not have one
 *         cell type per element of the grid
 */
template<std::floating_point T, std::size_t d>
std::vector<int> ghost_faces(const TensorGrid<T, d>& grid,
                             const CellClassification<T, d>& classification,
                             GhostFaces faces = GhostFaces::all);

/**
 * @brief Interior faces of a grid across which the volume fraction of the
 *        cells may jump
 *
 * @details Used in the Generalized Shifted Boundary Method to impose the
 *          skeleton conditions on faces with non-zero volume fraction jump
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

#endif // IGUANA_EMBEDDING_FACE_SELECTION_HPP
