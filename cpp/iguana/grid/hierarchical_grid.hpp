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

    /**
     * @brief Elements of a level in its domain, the region refined at least
     *        to that level: its active elements and the parents of those of
     *        the domain of the next level, in increasing index of the level
     *        grid
     *
     * @details The domains are nested, each a union of elements of the
     *          level before, and the functions of a hierarchical basis are
     *          selected by the domains their supports lie in
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels())
     */
    constexpr const std::vector<int>& domain(int level) const noexcept
    { return domains_[static_cast<std::size_t>(level)]; }

    /// @brief Number of active elements over all levels
    constexpr int num_elements() const noexcept
    { return offsets_.back(); }

    /**
     * @brief Index of the first active element of a level, that of the
     *        level after the last one being num_elements()
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels()]
     */
    constexpr int offset(int level) const noexcept
    { return offsets_[static_cast<std::size_t>(level)]; }

    /**
     * @brief Level of an active element
     *
     * @param element Element index
     *
     * @pre @p element lies in [0, num_elements())
     */
    int level_of(int element) const noexcept;

    /**
     * @brief Index of an active element in the grid of its level
     *
     * @param element Element index
     *
     * @pre @p element lies in [0, num_elements())
     */
    int level_index(int element) const noexcept;

    /**
     * @brief Parameters at which an active element starts
     *
     * @param element Element index
     *
     * @pre @p element lies in [0, num_elements())
     */
    std::array<T, d> element_start(int element) const noexcept;

    /**
     * @brief Parameters at which an active element ends
     *
     * @param element Element index
     *
     * @pre @p element lies in [0, num_elements())
     */
    std::array<T, d> element_end(int element) const noexcept;

    /// @brief Iterator at the first active element
    HierarchicalGridIterator<T, d> begin() const noexcept;

    /// @brief Sentinel past the last active element
    constexpr std::default_sentinel_t end() const noexcept
    { return std::default_sentinel; }

    /// @brief Whether two grids have the same levels and active elements
    bool operator==(const HierarchicalGrid& other) const = default;

private:
    template<std::floating_point U, std::size_t e>
    friend HierarchicalGrid<U, e> refine(const HierarchicalGrid<U, e>&,
                                           std::span<const int>);

    /// @brief Numbers the active elements and finds the domain of each
    ///        level from the active elements of the levels
    void update();

    /// @brief Tensor grid of each level
    std::vector<TensorGrid<T, d>> levels_;

    /// @brief Active elements of each level, in increasing index
    std::vector<std::vector<int>> active_;

    /// @brief Elements of each level in its domain, in increasing index
    std::vector<std::vector<int>> domains_;

    /// @brief Index of the first active element of each level, followed by
    ///        the number of active elements
    std::vector<int> offsets_;
};

/**
 * @brief Whether two hierarchical grids have the same elements, whatever
 *        the degrees and the multiplicities of their knots
 *
 * @param a First grid
 * @param b Second grid
 * @return Whether the knot lines of the coarse grids coincide and the same
 *         elements are active on every level
 */
template<std::floating_point T, std::size_t d>
bool same_elements(const HierarchicalGrid<T, d>& a,
                   const HierarchicalGrid<T, d>& b);

/**
 * @brief Active element holding a point, found level by level in the
 *        element of each level grid holding it
 *
 * @param grid Grid
 * @param parameters Parameters of the point, one per direction
 * @return Element index in [0, grid.num_elements())
 *
 * @pre @p parameters lie in the parametric domain of @p grid
 */
template<std::floating_point T, std::size_t d>
int element_holding(const HierarchicalGrid<T, d>& grid,
                    const std::array<T, d>& parameters);

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

/**
 * @brief Hierarchical grid refined by boxes of elements and graded, so that
 *        the hierarchical spline spaces on it keep their structure
 *
 * Each marked element is replaced by its children, along with the other
 * elements of the box of its level that holds it when refining by boxes.
 * The boxes of a level tile its grid from its first element, with q = p + 2
 * elements along a direction of degree p, the last box cut short at the
 * boundary, and the children of a box are the boxes of the next level.
 * Refining by boxes keeps the domain of each level a union of the supports
 * of the functions of degree p + 1 on the level before it, so that the
 * hierarchical spaces of degrees p + 1 and p form an exact sequence under
 * the divergence, with a box cut short at the boundary still the support
 * of a function of a clamped knot vector. Without boxes, the divergence of
 * the space of degree p + 1 still lies in that of degree p, which keeps a
 * divergence-conforming velocity divergence-free, but may not reach every
 * pressure
 *
 * The grid is then graded: whenever an element of a level lies within
 * p + 1 elements of a refined element of the level before, in each
 * direction of degree p, it is refined to that level as well, by its box
 * when refining by boxes. So the functions of the level before an active
 * element that are non-zero on its parent have their supports in the
 * domain of that level, and the truncated hierarchical functions non-zero
 * on the element come from at most two consecutive levels
 *
 * @param grid Grid to refine
 * @param elements Active elements to refine, in any order and possibly
 *        repeated
 * @param boxes Whether to refine the boxes holding the elements rather
 *        than the elements alone
 * @return Grid with the children of the elements, or of their boxes,
 *         active instead, graded
 *
 * @throws std::invalid_argument As refine() does
 */
template<std::floating_point T, std::size_t d>
HierarchicalGrid<T, d> refine_admissible(const HierarchicalGrid<T, d>& grid,
                                         std::span<const int> elements,
                                         bool boxes);

} // namespace iguana

#endif // IGUANA_GRID_HIERARCHICAL_GRID_HPP
