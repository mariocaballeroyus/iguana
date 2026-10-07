/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_SURROGATE_BOUNDARY_HPP
#define IGUANA_EMBEDDING_SURROGATE_BOUNDARY_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "iguana/embedding/embedded_domain.hpp"
#include "iguana/grid/tensor_grid.hpp"

namespace iguana
{

/**
 * @brief Boundary of the inside cells of a domain, the surrogate boundary
 *        of a shifted boundary method, divided over the elements of a grid
 *
 * The inside cells form the surrogate domain, which a shifted boundary
 * method integrates in place of the physical one. Its boundary runs along
 * knot lines, over the faces between an inside cell and a cell that is not
 * inside or lies past the edge of the grid, holes included. Each face is
 * held by its inside cell
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class SurrogateBoundary
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /// @brief Face of an inside cell along which the boundary runs, a box of
    ///        parameter space flat in its direction
    struct Face
    {
        /// @brief Direction normal to the face
        int direction;

        /// @brief Side of the cell, -1 at its start and 1 at its end, the
        ///        sign of the normal pointing out of the surrogate domain
        int side;

        /// @brief Corner of the face where every parameter is lowest
        std::array<T, d> start;

        /// @brief Corner of the face where every parameter is highest,
        ///        equal to start in the direction of the face
        std::array<T, d> end;
    };

    /**
     * @brief Finds the faces between the inside cells of a domain and the
     *        rest
     *
     * @param grid Grid whose elements are the cells
     * @param domain Cell type of each element of the grid
     *
     * @throws std::invalid_argument If the domain does not have one cell type
     *         per element of the grid
     */
    SurrogateBoundary(const TensorGrid<T, d>& grid,
                      const EmbeddedDomain<T, d>& domain);

    /// @brief Number of elements of the grid
    constexpr int num_elements() const noexcept
    { return static_cast<int>(offsets_.size()) - 1; }

    /// @brief Number of faces over all elements
    constexpr int num_faces() const noexcept
    { return static_cast<int>(faces_.size()); }

    /**
     * @brief Faces of the boundary held by an element, ordered by direction
     *        and the start side before the end one
     *
     * @param element Element index, in the numbering of the grid
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr std::span<const Face> faces_on_element(int element) const
        noexcept
    {
        const int first = offsets_(element);
        const int count = offsets_(element + 1) - first;

        return {faces_.data() + first, static_cast<std::size_t>(count)};
    }

private:
    /// @brief First face of each element, followed by the number of faces
    Eigen::VectorXi offsets_;

    /// @brief Faces, element after element
    std::vector<Face> faces_;
};

} // namespace iguana

#endif // IGUANA_EMBEDDING_SURROGATE_BOUNDARY_HPP
