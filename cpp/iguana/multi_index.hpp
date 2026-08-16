/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_MULTI_INDEX_HPP
#define IGUANA_MULTI_INDEX_HPP

#include<array>
#include<cstddef>

namespace iguana
{

/**
 * @brief Advances a multi-index, the first direction running fastest.
 *
 * The multi-index is incremented as an odometer, each direction carrying
 * into the next once it reaches its bound. From zeros, the successive
 * calls enumerate every multi-index below the bounds exactly once.
 *
 * @tparam d Number of directions.
 *
 * @param index The multi-index, advanced in place.
 * @param bounds The upper bound of each direction.
 *
 * @return True while a multi-index remains, false once the last one has
 *         been passed, in which case @p index is left at zeros.
 *
 * @pre @p index lies below @p bounds. It is not checked.
 */
template<std::size_t d>
bool next_lexicographic(std::array<int, d>& index,
                        const std::array<int, d>& bounds) noexcept
{
    for (std::size_t k = 0; k < d; ++k) {
        if (++index[k] < bounds[k])
            return true;

        // The direction is exhausted, so it carries into the next one
        index[k] = 0;
    }

    return false;
}

} // namespace iguana

#endif // IGUANA_MULTI_INDEX_HPP
