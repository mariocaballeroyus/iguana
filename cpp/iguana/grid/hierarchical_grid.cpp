/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "hierarchical_grid.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "iguana/grid/knot_vector.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/// @brief Knot vector with the midpoint of every element inserted
template<std::floating_point T>
KnotVector<T> halve_elements(const KnotVector<T>& knots)
{
    std::vector<T> midpoints(knots.num_elements());

    for (int element = 0; element < knots.num_elements(); ++element)
        midpoints[element] =
            (knots.element_start(element) + knots.element_end(element)) / 2;

    return insert_knots(knots, midpoints);
}

/// @brief Grid of the next level, with every element halved in each
///        direction
template<std::floating_point T, std::size_t d, std::size_t... direction>
TensorGrid<T, d> next_level(const TensorGrid<T, d>& grid,
                              std::index_sequence<direction...>)
{
    return TensorGrid<T, d>(std::array<KnotVector<T>, d>{
        halve_elements(grid.knots(direction))...});
}

/// @brief Number of elements of each direction of a grid
template<std::floating_point T, std::size_t d>
std::array<int, d> element_counts(const TensorGrid<T, d>& grid) noexcept
{
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] = grid.knots(direction).num_elements();

    return counts;
}

} // namespace

template<std::floating_point T, std::size_t d>
HierarchicalGrid<T, d>::HierarchicalGrid(TensorGrid<T, d> coarse)
    : active_(1)
{
    active_[0].resize(coarse.num_elements());
    std::iota(active_[0].begin(), active_[0].end(), 0);
    levels_.push_back(std::move(coarse));
    update();
}

template<std::floating_point T, std::size_t d>
int HierarchicalGrid<T, d>::level_of(int element) const noexcept
{
    // The offsets are non-decreasing, so the last one not above the element
    // is that of its level, skipping the levels without elements
    const auto after = std::ranges::upper_bound(offsets_, element);

    return static_cast<int>(std::distance(offsets_.begin(), after)) - 1;
}

template<std::floating_point T, std::size_t d>
int HierarchicalGrid<T, d>::level_index(int element) const noexcept
{
    const int level = level_of(element);

    return active_elements(level)[static_cast<std::size_t>(
        element - offset(level))];
}

template<std::floating_point T, std::size_t d>
std::array<T, d> HierarchicalGrid<T, d>::element_start(int element) const
    noexcept
{
    return level(level_of(element)).element_start(level_index(element));
}

template<std::floating_point T, std::size_t d>
std::array<T, d> HierarchicalGrid<T, d>::element_end(int element) const
    noexcept
{
    return level(level_of(element)).element_end(level_index(element));
}

template<std::floating_point T, std::size_t d>
HierarchicalGridIterator<T, d>
HierarchicalGrid<T, d>::begin() const noexcept
{
    return HierarchicalGridIterator<T, d>(*this);
}

template<std::floating_point T, std::size_t d>
void HierarchicalGrid<T, d>::update()
{
    offsets_.assign(1, 0);

    for (const std::vector<int>& active : active_)
        offsets_.push_back(offsets_.back() + static_cast<int>(active.size()));

    // The domain of the finest level is its active elements, and that of
    // each level before adds the parents, i / 2 in each direction, of the
    // domain of the next
    domains_.assign(active_.size(), {});
    domains_.back() = active_.back();

    for (std::size_t level = active_.size() - 1; level-- > 0;) {
        const std::array<int, d> counts = element_counts(levels_[level]);
        const std::array<int, d> fine_counts =
            element_counts(levels_[level + 1]);
        std::vector<int>& domain = domains_[level];
        domain = active_[level];

        for (const int child : domains_[level + 1]) {
            std::array<int, d> parent = unflatten(child, fine_counts);

            for (int& index : parent)
                index /= 2;

            domain.push_back(flatten(parent, counts));
        }

        std::ranges::sort(domain);
        const auto repeats = std::ranges::unique(domain);
        domain.erase(repeats.begin(), repeats.end());
    }
}

template<std::floating_point T, std::size_t d>
HierarchicalGrid<T, d> refine(const HierarchicalGrid<T, d>& grid,
                                std::span<const int> elements)
{
    const int num_elements = grid.num_elements();

    if (!std::ranges::all_of(elements, [num_elements](int element) {
            return 0 <= element && element < num_elements;
        }))
        throw std::invalid_argument("refine: "
                                    "the elements must lie in the grid");

    // Marked elements of each level, by their index in the level grid.
    // The offsets are non-decreasing, so the last one not above an element
    // is that of its level, skipping the levels without elements
    std::vector<std::vector<int>> marked(grid.num_levels());

    for (const int element : elements)
        marked[grid.level_of(element)].push_back(grid.level_index(element));

    HierarchicalGrid<T, d> refined = grid;

    for (int level = 0; level < grid.num_levels(); ++level) {
        std::vector<int>& parents = marked[level];

        if (parents.empty())
            continue;

        std::ranges::sort(parents);
        const auto repeats = std::ranges::unique(parents);
        parents.erase(repeats.begin(), repeats.end());

        std::vector<int> kept;
        std::ranges::set_difference(refined.active_[level], parents,
                                    std::back_inserter(kept));
        refined.active_[level] = std::move(kept);

        if (level + 1 == refined.num_levels()) {
            // The new level has 2^d times as many elements, which must stay
            // countable by int
            constexpr int limit = std::numeric_limits<int>::max() >> d;

            if (refined.levels_[level].num_elements() > limit)
                throw std::invalid_argument("refine: "
                                            "the next level would have too "
                                            "many elements");

            refined.levels_.push_back(
                next_level(refined.levels_[level],
                           std::make_index_sequence<d>{}));
            refined.active_.emplace_back();
        }

        // Each element of the next level halves one of this level in each
        // direction, so the children of element i are 2i + {0, 1}^d
        const std::array<int, d> bounds =
            element_counts(refined.levels_[level]);
        std::array<int, d> fine_bounds = bounds;

        for (int& bound : fine_bounds)
            bound *= 2;

        std::array<int, d> twos{};
        twos.fill(2);

        std::vector<int>& children = refined.active_[level + 1];

        for (const int parent : parents) {
            const std::array<int, d> origin = unflatten(parent, bounds);
            std::array<int, d> offset{};

            do {
                std::array<int, d> child{};

                for (std::size_t direction = 0; direction < d; ++direction)
                    child[direction] =
                        2 * origin[direction] + offset[direction];

                children.push_back(flatten(child, fine_bounds));
            } while (next_lexicographic(offset, twos));
        }

        // The children lie in elements that were active, so none of them
        // is active yet
        std::ranges::sort(children);
    }

    refined.update();

    return refined;
}

namespace
{

/// @brief Whether a sorted vector holds a value
bool holds(const std::vector<int>& sorted, int value) noexcept
{
    return std::ranges::binary_search(sorted, value);
}

/**
 * @brief Active element covering an element of a level grid, the element
 *        itself if active, or else its nearest active ancestor
 *
 * @return Hierarchical index of the active element
 *
 * @pre The element lies in the domain of its level, or in that of a level
 *      before it
 */
template<std::floating_point T, std::size_t d>
int covering(const HierarchicalGrid<T, d>& grid, int level, int element)
{
    for (;; --level) {
        const std::vector<int>& active = grid.active_elements(level);
        const auto found = std::ranges::lower_bound(active, element);

        if (found != active.end() && *found == element)
            return grid.offset(level)
                   + static_cast<int>(std::distance(active.begin(), found));

        // The parent, i / 2 in each direction
        std::array<int, d> index =
            unflatten(element, element_counts(grid.level(level)));

        for (int& axis : index)
            axis /= 2;

        element = flatten(index, element_counts(grid.level(level - 1)));
    }
}

/**
 * @brief Active elements to refine so that the boxes holding the marked
 *        elements are refined, the box of a level holding an element being
 *        the aligned q elements along each direction, q its degree plus two
 *        when expanded and the element alone otherwise
 *
 * An element of a box that a coarser active element covers marks that
 * element in turn, whose own box is taken, so that the result is a union
 * of boxes of every level marked
 *
 * @return Hierarchical indices, sorted and unique
 */
template<std::floating_point T, std::size_t d>
std::vector<int> boxes(const HierarchicalGrid<T, d>& grid,
                       std::vector<int> marked, bool expanded)
{
    std::vector<int> result;

    while (!marked.empty()) {
        const int element = marked.back();
        marked.pop_back();

        const int level = grid.level_of(element);
        const std::array<int, d> counts = element_counts(grid.level(level));
        const std::array<int, d> index =
            unflatten(grid.level_index(element), counts);

        // The box, cut short at the boundary
        std::array<int, d> start{};
        std::array<int, d> extent{};

        for (std::size_t direction = 0; direction < d; ++direction) {
            const int q =
                expanded ? grid.level(0).knots(direction).degree() + 2 : 1;
            start[direction] = index[direction] / q * q;
            extent[direction] =
                std::min(q, counts[direction] - start[direction]);
        }

        std::array<int, d> offset{};

        do {
            std::array<int, d> other = start;

            for (std::size_t direction = 0; direction < d; ++direction)
                other[direction] += offset[direction];

            const int flat = flatten(other, counts);

            // An element of the box refined already needs nothing, one
            // active joins the result, and one covered by a coarser active
            // element marks that one in turn
            if (holds(grid.domain(level), flat)) {
                if (holds(grid.active_elements(level), flat))
                    result.push_back(covering(grid, level, flat));
            } else {
                marked.push_back(covering(grid, level, flat));
            }
        } while (next_lexicographic(offset, extent));
    }

    std::ranges::sort(result);
    const auto repeats = std::ranges::unique(result);
    result.erase(repeats.begin(), repeats.end());

    return result;
}

/**
 * @brief Active elements whose refinement grades the grid, those covering
 *        the elements within p + 1 elements of the parent of an active
 *        element, in each direction of degree p, that its level does not
 *        refine
 *
 * @return Hierarchical indices, in any order and possibly repeated
 */
template<std::floating_point T, std::size_t d>
std::vector<int> violations(const HierarchicalGrid<T, d>& grid)
{
    std::vector<int> marked;

    for (int level = 2; level < grid.num_levels(); ++level) {
        const int before = level - 1;
        const std::array<int, d> counts = element_counts(grid.level(level));
        const std::array<int, d> coarse_counts =
            element_counts(grid.level(before));

        for (const int element : grid.active_elements(level)) {
            std::array<int, d> parent = unflatten(element, counts);
            std::array<int, d> start{};
            std::array<int, d> extent{};

            for (std::size_t direction = 0; direction < d; ++direction) {
                const int reach =
                    grid.level(0).knots(direction).degree() + 1;
                parent[direction] /= 2;
                start[direction] = std::max(0, parent[direction] - reach);
                extent[direction] =
                    std::min(coarse_counts[direction],
                             parent[direction] + reach + 1)
                    - start[direction];
            }

            std::array<int, d> offset{};

            do {
                std::array<int, d> other = start;

                for (std::size_t direction = 0; direction < d; ++direction)
                    other[direction] += offset[direction];

                const int flat = flatten(other, coarse_counts);

                if (!holds(grid.domain(before), flat))
                    marked.push_back(covering(grid, before, flat));
            } while (next_lexicographic(offset, extent));
        }
    }

    return marked;
}

} // namespace

template<std::floating_point T, std::size_t d>
HierarchicalGrid<T, d> refine_admissible(const HierarchicalGrid<T, d>& grid,
                                         std::span<const int> elements,
                                         bool boxes)
{
    // Before the boxes read the elements
    const int num_elements = grid.num_elements();

    if (!std::ranges::all_of(elements, [num_elements](int element) {
            return 0 <= element && element < num_elements;
        }))
        throw std::invalid_argument("refine_admissible: "
                                    "the elements must lie in the grid");

    HierarchicalGrid<T, d> refined = grid;
    std::vector<int> marked =
        iguana::boxes(grid, {elements.begin(), elements.end()}, boxes);

    // Grading refines levels before the finest one only, so it ends
    while (!marked.empty()) {
        refined = refine(refined, marked);
        marked = iguana::boxes(refined, violations(refined), boxes);
    }

    return refined;
}

template<std::floating_point T, std::size_t d>
bool same_elements(const HierarchicalGrid<T, d>& a,
                   const HierarchicalGrid<T, d>& b)
{
    if (a.num_levels() != b.num_levels()
        || !same_elements(a.level(0), b.level(0)))
        return false;

    for (int level = 0; level < a.num_levels(); ++level) {
        if (a.active_elements(level) != b.active_elements(level))
            return false;
    }

    return true;
}

template<std::floating_point T, std::size_t d>
int element_holding(const HierarchicalGrid<T, d>& grid,
                    const std::array<T, d>& parameters)
{
    // The element holding the point on each level, which is active on the
    // first level whose domain does not go on refining it
    for (int level = 0;; ++level) {
        const std::vector<int>& active = grid.active_elements(level);
        const int element = element_holding(grid.level(level), parameters);
        const auto found = std::ranges::lower_bound(active, element);

        if (found != active.end() && *found == element)
            return grid.offset(level)
                   + static_cast<int>(std::distance(active.begin(), found));
    }
}

template class HierarchicalGrid<double, 1>;
template class HierarchicalGrid<double, 2>;
template class HierarchicalGrid<double, 3>;

template bool same_elements(const HierarchicalGrid<double, 1>&,
                            const HierarchicalGrid<double, 1>&);
template bool same_elements(const HierarchicalGrid<double, 2>&,
                            const HierarchicalGrid<double, 2>&);
template bool same_elements(const HierarchicalGrid<double, 3>&,
                            const HierarchicalGrid<double, 3>&);

template int element_holding(const HierarchicalGrid<double, 1>&,
                             const std::array<double, 1>&);
template int element_holding(const HierarchicalGrid<double, 2>&,
                             const std::array<double, 2>&);
template int element_holding(const HierarchicalGrid<double, 3>&,
                             const std::array<double, 3>&);

template HierarchicalGrid<double, 1> refine(
    const HierarchicalGrid<double, 1>&, std::span<const int>);
template HierarchicalGrid<double, 2> refine(
    const HierarchicalGrid<double, 2>&, std::span<const int>);
template HierarchicalGrid<double, 3> refine(
    const HierarchicalGrid<double, 3>&, std::span<const int>);

template HierarchicalGrid<double, 1> refine_admissible(
    const HierarchicalGrid<double, 1>&, std::span<const int>, bool);
template HierarchicalGrid<double, 2> refine_admissible(
    const HierarchicalGrid<double, 2>&, std::span<const int>, bool);
template HierarchicalGrid<double, 3> refine_admissible(
    const HierarchicalGrid<double, 3>&, std::span<const int>, bool);

} // namespace iguana
