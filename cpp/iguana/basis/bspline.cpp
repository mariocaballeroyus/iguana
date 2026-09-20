/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "bspline.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace iguana
{

template<std::floating_point T>
BSpline<T>::BSpline(int degree, std::vector<T> knots)
    : degree_(degree), knots_(std::move(knots))
{
    if (degree_ < 0)
        throw std::invalid_argument("BSpline: "
                                    "the degree must be non-negative");

    if (degree_ > max_degree)
        throw std::invalid_argument("BSpline: "
                                    "the degree exceeds max_degree");

    if (!std::ranges::is_sorted(knots_))
        throw std::invalid_argument("BSpline: "
                                    "the knots must be non-decreasing");

    const std::size_t min_knots =
        2 * (static_cast<std::size_t>(degree_) + 1);

    if (knots_.size() < min_knots)
        throw std::invalid_argument("BSpline: "
                                    "too few knots for the given degree");

    const int num_functions =
        static_cast<int>(knots_.size()) - degree_ - 1;

    for (int span = degree_; span < num_functions; ++span) {
        if (knots_[span] < knots_[span + 1])
            element_spans_.push_back(span);
    }

    if (element_spans_.empty())
        throw std::invalid_argument("BSpline: "
                                    "the parametric domain must be non-empty");
}

template class BSpline<double>;

} // namespace iguana
