/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_JUMP_FACES_HPP
#define IGUANA_EMBEDDING_JUMP_FACES_HPP

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
 */
template<std::floating_point T, std::size_t d>
class JumpFaces
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
     * @brief Finds the faces of a grid across which the volume fraction
     *        may jump
     *
     * @param grid Grid whose elements are the cells
     * @param classification Cell type of each element of the grid
     *
     * @throws std::invalid_argument If the classification does not have one
     *         cell type per element of the grid
     */
    JumpFaces(const TensorGrid<T, d>& grid,
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

#endif // IGUANA_EMBEDDING_JUMP_FACES_HPP
