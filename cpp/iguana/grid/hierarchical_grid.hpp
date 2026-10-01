/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GRID_HIERARCHICAL_GRID_HPP
#define IGUANA_GRID_HIERARCHICAL_GRID_HPP

#include <concepts>
#include <cstddef>
#include <iterator>
#include <span>
#include <vector>

#include "iguana/grid/hierarchical_grid_iterator.hpp"
#include "iguana/grid/tensor_grid.hpp"

namespace iguana
{

/**
 * @brief Elements of a hierarchical basis, active elements taken from a
 *        sequence of nested tensor grids
 *
 * Level 0 is the coarse tensor grid, and level l + 1 halves every
 * element of level l in each direction, so that element i of level l has
 * the 2^d children 2i + {0, 1}^d. The active elements of the levels tile
 * the parametric domain. They are numbered level by level and, within a
 * level, in increasing index of the level grid
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class HierarchicalGrid
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the grid with a single level, whose elements are
     *        all active
     *
     * @param coarse Tensor grid of level 0
     */
    explicit HierarchicalGrid(TensorGrid<T, d> coarse);

    /// @brief Number of levels, the finest one possibly without elements
    constexpr int num_levels() const noexcept
    { return static_cast<int>(levels_.size()); }

    /**
     * @brief Tensor grid of a level
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels())
     */
    constexpr const TensorGrid<T, d>& level(int level) const noexcept
    { return levels_[static_cast<std::size_t>(level)]; }

    /**
     * @brief Active elements of a level, in increasing index of the level
     *        grid
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels())
     */
    constexpr const std::vector<int>& active_elements(int level) const noexcept
    { return active_[static_cast<std::size_t>(level)]; }

    /// @brief Number of active elements over all levels
    constexpr int num_elements() const noexcept
    { return offsets_.back(); }

    /// @brief Iterator at the first active element
    HierarchicalGridIterator<T, d> begin() const noexcept;

    /// @brief Sentinel past the last active element
    constexpr std::default_sentinel_t end() const noexcept
    { return std::default_sentinel; }

private:
    template<std::floating_point U, std::size_t e>
    friend HierarchicalGrid<U, e> refine(const HierarchicalGrid<U, e>&,
                                           std::span<const int>);

    /// @brief Tensor grid of each level
    std::vector<TensorGrid<T, d>> levels_;

    /// @brief Active elements of each level, in increasing index
    std::vector<std::vector<int>> active_;

    /// @brief Index of the first active element of each level, followed by
    ///        the number of active elements
    std::vector<int> offsets_;
};

/**
 * @brief Hierarchical grid with elements replaced by their children
 *
 * Each marked element is replaced by its 2^d children on the next level,
 * which is added if the element lies on the finest one. The other
 * elements stay active, although the numbering changes
 *
 * @param grid Grid to refine
 * @param elements Active elements to refine, in any order and possibly
 *        repeated
 * @return Grid with the children of the marked elements active instead
 *
 * @throws std::invalid_argument If an element lies outside
 *         [0, grid.num_elements()), or if a new level would have more
 *         elements than int can count
 */
template<std::floating_point T, std::size_t d>
HierarchicalGrid<T, d> refine(const HierarchicalGrid<T, d>& grid,
                                std::span<const int> elements);

} // namespace iguana

#endif // IGUANA_GRID_HIERARCHICAL_GRID_HPP
