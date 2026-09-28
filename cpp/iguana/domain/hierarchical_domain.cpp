/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "hierarchical_domain.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "iguana/domain/knot_vector.hpp"
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

/// @brief Domain of the next level, with every element halved in each
///        direction
template<std::floating_point T, std::size_t d, std::size_t... direction>
TensorDomain<T, d> next_level(const TensorDomain<T, d>& domain,
                              std::index_sequence<direction...>)
{
    return TensorDomain<T, d>(std::array<KnotVector<T>, d>{
        halve_elements(domain.knots(direction))...});
}

} // namespace

template<std::floating_point T, std::size_t d>
HierarchicalDomain<T, d>::HierarchicalDomain(TensorDomain<T, d> coarse)
    : active_(1),
      offsets_{0, coarse.num_elements()}
{
    active_[0].resize(coarse.num_elements());
    std::iota(active_[0].begin(), active_[0].end(), 0);
    levels_.push_back(std::move(coarse));
}

template<std::floating_point T, std::size_t d>
HierarchicalDomain<T, d> refine(const HierarchicalDomain<T, d>& domain,
                                std::span<const int> elements)
{
    const int num_elements = domain.num_elements();

    if (!std::ranges::all_of(elements, [num_elements](int element) {
            return 0 <= element && element < num_elements;
        }))
        throw std::invalid_argument("refine: "
                                    "the elements must lie in the domain");

    // Marked elements of each level, by their index in the level domain.
    // The offsets are non-decreasing, so the last one not above an element
    // is that of its level, skipping the levels without elements
    std::vector<std::vector<int>> marked(domain.num_levels());

    for (const int element : elements) {
        const auto after = std::ranges::upper_bound(domain.offsets_, element);
        const int level =
            static_cast<int>(std::distance(domain.offsets_.begin(), after)) - 1;

        marked[level].push_back(
            domain.active_[level][element - domain.offsets_[level]]);
    }

    HierarchicalDomain<T, d> refined = domain;

    for (int level = 0; level < domain.num_levels(); ++level) {
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
        std::array<int, d> bounds{};
        std::array<int, d> fine_bounds{};

        for (std::size_t direction = 0; direction < d; ++direction) {
            bounds[direction] =
                refined.levels_[level].knots(direction).num_elements();
            fine_bounds[direction] = 2 * bounds[direction];
        }

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

    refined.offsets_.assign(1, 0);

    for (const std::vector<int>& active : refined.active_)
        refined.offsets_.push_back(refined.offsets_.back() +
                                   static_cast<int>(active.size()));

    return refined;
}

template class HierarchicalDomain<double, 1>;
template class HierarchicalDomain<double, 2>;
template class HierarchicalDomain<double, 3>;

template HierarchicalDomain<double, 1> refine(
    const HierarchicalDomain<double, 1>&, std::span<const int>);
template HierarchicalDomain<double, 2> refine(
    const HierarchicalDomain<double, 2>&, std::span<const int>);
template HierarchicalDomain<double, 3> refine(
    const HierarchicalDomain<double, 3>&, std::span<const int>);

} // namespace iguana
