/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_GHOST_FACES_HPP
#define IGUANA_EMBEDDING_GHOST_FACES_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
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
 */
template<std::floating_point T, std::size_t d>
class GhostFaces
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /// @brief Face between two neighboring cells, a box of parameter space
    ///        flat in its direction
    struct Face
    {
        /// @brief Direction normal to the face
        int direction;

        /// @brief Cell before the face along its direction, in the numbering
        ///        of the grid
        int before;

        /// @brief Cell after the face along its direction, in the numbering
        ///        of the grid
        int after;

        /// @brief Corner of the face where every parameter is lowest
        std::array<T, d> start;

        /// @brief Corner of the face where every parameter is highest,
        ///        equal to start in the direction of the face
        std::array<T, d> end;
    };

    /**
     * @brief Finds the faces of a grid where a ghost penalty acts
     *
     * @param grid Grid whose elements are the cells
     * @param classification Cell type of each element of the grid
     *
     * @throws std::invalid_argument If the classification does not have one
     *         cell type per element of the grid
     */
    GhostFaces(const TensorGrid<T, d>& grid,
               const CellClassification<T, d>& classification);

    /// @brief Number of faces
    constexpr int num_faces() const noexcept
    { return static_cast<int>(faces_.size()); }

    /// @brief Faces, ordered by the cell before them and then by direction
    constexpr std::span<const Face> faces() const noexcept
    { return faces_; }

private:
    /// @brief Faces, ordered by the cell before them and then by direction
    std::vector<Face> faces_;
};

} // namespace iguana

#endif // IGUANA_EMBEDDING_GHOST_FACES_HPP
