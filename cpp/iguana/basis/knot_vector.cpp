/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "knot_vector.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace iguana
{

template<std::floating_point T>
KnotVector<T>::KnotVector(int degree, std::vector<T> knots)
    : degree_(degree), knots_(std::move(knots))
{
    if (degree_ < 0)
        throw std::invalid_argument("KnotVector: "
                                    "the degree must be non-negative");

    if (!std::ranges::is_sorted(knots_))
        throw std::invalid_argument("KnotVector: "
                                    "the knots must be non-decreasing");

    const std::size_t min_knots =
        2 * (static_cast<std::size_t>(degree_) + 1);

    if (knots_.size() < min_knots)
        throw std::invalid_argument("KnotVector: "
                                    "too few knots for the given degree");

    // The domain runs from the knot of index p to the one of index
    // num_knots - p - 1, and its elements are the non-empty spans between
    const int last = static_cast<int>(knots_.size()) - degree_ - 1;

    for (int span = degree_; span < last; ++span) {
        if (knots_[span] < knots_[span + 1])
            element_spans_.push_back(span);
    }

    if (element_spans_.empty())
        throw std::invalid_argument("KnotVector: "
                                    "the parametric domain must be "
                                    "non-empty");
}

template class KnotVector<double>;

} // namespace iguana
