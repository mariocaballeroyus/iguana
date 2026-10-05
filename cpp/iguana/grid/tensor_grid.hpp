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
 * The elements are numbered with the first direction running fastest
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
    /// @brief Knot vector of each parametric direction
    std::array<KnotVector<T>, d> knots_;

    /// @brief Product of the element counts of each direction
    int num_elements_;
};

} // namespace iguana

#endif // IGUANA_GRID_TENSOR_GRID_HPP
