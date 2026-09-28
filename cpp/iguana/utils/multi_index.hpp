/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_UTILS_MULTI_INDEX_HPP
#define IGUANA_UTILS_MULTI_INDEX_HPP

#include <array>
#include <cstddef>

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

} // namespace iguana

#endif // IGUANA_UTILS_MULTI_INDEX_HPP
