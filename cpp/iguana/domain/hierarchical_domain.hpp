/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_DOMAIN_HIERARCHICAL_DOMAIN_HPP
#define IGUANA_DOMAIN_HIERARCHICAL_DOMAIN_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "iguana/domain/tensor_domain.hpp"

namespace iguana
{

/**
 * @brief Elements of a hierarchical basis, active elements taken from a
 *        sequence of nested tensor domains
 *
 * Level 0 is the coarse tensor domain, and level l + 1 halves every
 * element of level l in each direction, so that element i of level l has
 * the 2^d children 2i + {0, 1}^d. The active elements of the levels tile
 * the parametric domain. They are numbered level by level and, within a
 * level, in increasing index of the level domain
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class HierarchicalDomain
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the domain with a single level, whose elements are
     *        all active
     *
     * @param coarse Tensor domain of level 0
     */
    explicit HierarchicalDomain(TensorDomain<T, d> coarse);

    /// @brief Number of levels, the finest one possibly without elements
    constexpr int num_levels() const noexcept
    { return static_cast<int>(levels_.size()); }

    /**
     * @brief Tensor domain of a level
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels())
     */
    constexpr const TensorDomain<T, d>& level(int level) const noexcept
    { return levels_[static_cast<std::size_t>(level)]; }

    /**
     * @brief Active elements of a level, in increasing index of the level
     *        domain
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

private:
    /// @brief Tensor domain of each level
    std::vector<TensorDomain<T, d>> levels_;

    /// @brief Active elements of each level, in increasing index
    std::vector<std::vector<int>> active_;

    /// @brief Index of the first active element of each level, followed by
    ///        the number of active elements
    std::vector<int> offsets_;
};

} // namespace iguana

#endif // IGUANA_DOMAIN_HIERARCHICAL_DOMAIN_HPP
