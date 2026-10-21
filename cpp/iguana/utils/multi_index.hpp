/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_UTILS_MULTI_INDEX_HPP
#define IGUANA_UTILS_MULTI_INDEX_HPP

#include <array>
#include <cstddef>
#include <utility>

namespace iguana
{

/**
 * @brief Advances a multi-index with the first direction running fastest.
 *
 * @tparam d Number of directions.
 * @param index Multi-index to advance.
 * @param bounds Exclusive upper bound in each direction.
 * @return Whether another multi-index remains. If not, @p index is reset
 *         to zero in every direction.
 *
 * @pre Every entry of @p index lies in [0, bounds), and every bound is
 *      positive.
 */
template<std::size_t d>
bool next_lexicographic(std::array<int, d>& index,
                        const std::array<int, d>& bounds) noexcept
{
    for (std::size_t direction = 0; direction < d; ++direction) {
        if (++index[direction] < bounds[direction])
            return true;

        index[direction] = 0;
    }

    return false;
}

/**
 * @brief Flattens a multi-index with the first direction running fastest.
 *
 * @tparam d Number of directions.
 * @param index Multi-index to flatten.
 * @param bounds Exclusive upper bound in each direction.
 * @return Flat index below the product of @p bounds.
 *
 * @pre Every entry of @p index lies in [0, bounds), and every bound is
 *      positive.
 */
template<std::size_t d>
int flatten(const std::array<int, d>& index,
            const std::array<int, d>& bounds) noexcept
{
    int flat = 0;

    for (std::size_t direction = d; direction-- > 0;)
        flat = flat * bounds[direction] + index[direction];

    return flat;
}

/**
 * @brief Splits a flat index into a multi-index with the first direction
 *        running fastest, the inverse of flatten()
 *
 * @tparam d Number of directions
 * @param flat Flat index to split
 * @param bounds Exclusive upper bound in each direction
 * @return Multi-index with every entry in [0, bounds)
 *
 * @pre @p flat lies in [0, product of @p bounds), and every bound is
 *      positive
 */
template<std::size_t d>
std::array<int, d> unflatten(int flat,
                             const std::array<int, d>& bounds) noexcept
{
    std::array<int, d> index{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        index[direction] = flat % bounds[direction];
        flat /= bounds[direction];
    }

    return index;
}

/**
 * @brief Number of multi-indices of two directions of total order at most
 *        an order, (order + 1) (order + 2) / 2
 *
 * @param order Highest total order
 * @return Number of pairs (a, b) with a + b at most @p order
 *
 * @pre @p order is non-negative
 */
constexpr int simplex_size(int order) noexcept
{
    return (order + 1) * (order + 2) / 2;
}

/**
 * @brief Index of the second derivative along two directions, the pure ones
 *        first, then the mixed ones in lexicographic order of their
 *        directions: xx, yy, zz, xy, xz, yz
 *
 * @details The order of the Hessians of the bases, of the map and in
 *          physical space. A pair and its reverse share one index
 *
 * @tparam d Number of directions
 * @param a First direction
 * @param b Second direction
 * @return Index below d (d + 1) / 2
 *
 * @pre @p a and @p b lie in [0, d)
 */
template<std::size_t d>
constexpr std::size_t pair_index(std::size_t a, std::size_t b) noexcept
{
    if (a == b)
        return a;

    if (a > b)
        std::swap(a, b);

    // The mixed pairs of each first direction before a, then those of a
    std::size_t index = d;

    for (std::size_t first = 0; first < a; ++first)
        index += d - 1 - first;

    return index + b - a - 1;
}

} // namespace iguana

#endif // IGUANA_UTILS_MULTI_INDEX_HPP
