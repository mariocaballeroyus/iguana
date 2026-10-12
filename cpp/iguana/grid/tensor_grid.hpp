/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GRID_TENSOR_GRID_HPP
#define IGUANA_GRID_TENSOR_GRID_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <vector>

#include "iguana/grid/knot_vector.hpp"
#include "iguana/grid/tensor_grid_iterator.hpp"

namespace iguana
{

/**
 * @brief Elements of a tensor-product basis, the products of the elements
 *        of one knot vector per direction
 *
 * The elements are numbered with the first direction running fastest. The
 * interior faces, each between two neighboring elements, are numbered
 * direction by direction: first those normal to the first direction, and
 * among those normal to one direction, as the elements before them, which
 * stop one short of the last element along it
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class TensorGrid
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the grid from the knot vector of each direction
     *
     * @param knots Knot vector of each parametric direction
     */
    explicit TensorGrid(std::array<KnotVector<T>, d> knots);

    /**
     * @brief Knot vector of a parametric direction
     *
     * @param direction Direction index
     *
     * @pre @p direction lies in [0, dimension)
     */
    constexpr const KnotVector<T>& knots(std::size_t direction) const noexcept
    { return knots_[direction]; }

    /// @brief Number of elements, the product of those of each direction
    constexpr int num_elements() const noexcept
    { return num_elements_; }

    /// @brief Number of interior faces, each between two neighboring
    ///        elements
    constexpr int num_faces() const noexcept
    { return face_offsets_[d]; }

    /**
     * @brief Direction normal to an interior face
     *
     * @param face Face index
     *
     * @pre @p face lies in [0, num_faces())
     */
    std::size_t face_direction(int face) const noexcept;

    /**
     * @brief Elements on either side of an interior face: the element
     *        before it along its direction, then the element after it
     *
     * @param face Face index
     *
     * @pre @p face lies in [0, num_faces())
     */
    std::array<int, 2> face_elements(int face) const noexcept;

    /**
     * @brief Corner of an interior face where every parameter is lowest
     *
     * @param face Face index
     *
     * @pre @p face lies in [0, num_faces())
     */
    std::array<T, d> face_start(int face) const noexcept;

    /**
     * @brief Corner of an interior face where every parameter is highest,
     *        equal to its start in its direction
     *
     * @param face Face index
     *
     * @pre @p face lies in [0, num_faces())
     */
    std::array<T, d> face_end(int face) const noexcept;

    /**
     * @brief Coordinates of the knot lines, or planes, of each direction,
     *        the boundaries of its elements in increasing order
     *
     * Element e of a direction spans [lines[e], lines[e + 1]], so that a
     * repeated knot gives one line and the knots outside the domain of an
     * unclamped knot vector none
     */
    std::array<std::vector<T>, d> lines() const;

    /// @brief Iterator at the first element
    TensorGridIterator<T, d> begin() const noexcept;

    /// @brief Sentinel past the last element
    constexpr std::default_sentinel_t end() const noexcept
    { return std::default_sentinel; }

private:
    /// @brief Number of elements of each direction
    std::array<int, d> element_counts() const noexcept;

    /// @brief Multi-index of the element before an interior face
    std::array<int, d> element_before(int face) const noexcept;

    /// @brief Knot vector of each parametric direction
    std::array<KnotVector<T>, d> knots_;

    /// @brief Product of the element counts of each direction
    int num_elements_;

    /// @brief First face normal to each direction, followed by the number of
    ///        faces
    std::array<int, d + 1> face_offsets_{};
};

} // namespace iguana

#endif // IGUANA_GRID_TENSOR_GRID_HPP
