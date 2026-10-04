/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_EMBEDDED_BOUNDARY_HPP
#define IGUANA_EMBEDDING_EMBEDDED_BOUNDARY_HPP

#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "iguana/geometry/boundary.hpp"
#include "iguana/grid/tensor_grid.hpp"

namespace iguana
{

/**
 * @brief Boundary divided exactly over the elements of a background grid
 *
 * Each element of a face, in the parameter space of the grid, is split
 * where it crosses a knot line, so that each piece lies in one element of
 * its face and one of the grid. A piece lying on a knot line goes to the
 * element opposite its normal, and parts outside the grid are left out.
 * The pieces refer to the faces by index, and no geometry is held
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions, two
 */
template<std::floating_point T, std::size_t d>
class EmbeddedBoundary
{
    static_assert(d == 2, "EmbeddedBoundary: "
                          "the boundary must be made of curves");

public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /// @brief Width, relative to an element of a face, below which
    ///        crossings merge
    static constexpr T tolerance = 4096 * std::numeric_limits<T>::epsilon();

    /// @brief Part of a face inside one element of the face and one of the
    ///        grid, an interval of its parameter
    struct Piece
    {
        /// @brief Index of the face in the boundary
        int face;

        /// @brief Element of the face holding the piece
        int face_element;

        /// @brief Parameter of the face at which the piece starts
        T start;

        /// @brief Parameter of the face at which the piece ends
        T end;
    };

    /**
     * @brief Divides the faces of a boundary over the elements of a grid
     *
     * @param grid Grid whose elements divide the boundary
     * @param boundary Boundary whose faces lie in the parameter space of
     *        the grid
     *
     * @throws std::invalid_argument If the knot vector of a face is not
     *         clamped
     *
     * @pre A face lying on a knot line has exactly the coordinate of the
     *      line at its control points, and no face runs back and forth
     *      along it
     */
    EmbeddedBoundary(const TensorGrid<T, d>& grid,
                     const Boundary<T, d>& boundary);

    /// @brief Number of elements of the grid
    constexpr int num_elements() const noexcept
    { return static_cast<int>(offsets_.size()) - 1; }

    /// @brief Number of pieces over all elements
    constexpr int num_pieces() const noexcept
    { return static_cast<int>(pieces_.size()); }

    /**
     * @brief Pieces of the boundary inside an element, ordered by face, by
     *        element of the face and by parameter
     *
     * @param element Element index, in the numbering of the grid
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr std::span<const Piece> pieces_on_element(int element) const
        noexcept
    {
        const int first = offsets_(element);
        const int count = offsets_(element + 1) - first;
        return {pieces_.data() + first, static_cast<std::size_t>(count)};
    }

private:
    /// @brief First piece of each element, followed by the number of pieces
    Eigen::VectorXi offsets_;

    /// @brief Pieces, element after element
    std::vector<Piece> pieces_;
};

} // namespace iguana

#endif // IGUANA_EMBEDDING_EMBEDDED_BOUNDARY_HPP
