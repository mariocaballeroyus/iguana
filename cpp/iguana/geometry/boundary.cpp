/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t n>
Boundary<T, n>::Boundary(std::vector<Face> faces, std::vector<int> signs)
    : faces_(std::move(faces)),
      signs_(std::move(signs))
{
    if (signs_.size() != faces_.size())
        throw std::invalid_argument("Boundary: "
                                    "there must be one sign per face");

    if (!std::ranges::all_of(signs_, [](int sign) {
            return sign == 1 || sign == -1;
        }))
        throw std::invalid_argument("Boundary: "
                                    "the signs must be 1 or -1");
}

// Curves bounding planar regions and surfaces bounding volumes
template class Boundary<double, 2>;
template class Boundary<double, 3>;

} // namespace iguana
