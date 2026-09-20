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

template<std::floating_point T>
void BSpline<T>::eval_on_element(int first_active, std::span<const T> points,
                                 Eigen::MatrixX<T>& values) const
{
    const int deg = degree_;
    const Eigen::Index num_pts = static_cast<Eigen::Index>(points.size());
    const T* const knots = knots_.data();

    // Knot span shared by all evaluation points
    const int span = first_active + deg;

    // Each column holds the active values at one point
    values.resize(deg + 1, num_pts);

    // Stack storage bounded by max_degree
    T left[max_degree + 1];
    T right[max_degree + 1];

    for (Eigen::Index q = 0; q < num_pts; ++q) {
        const T u = points[static_cast<std::size_t>(q)];
        T* const n = values.col(q).data();

        // Piecewise-constant starting value
        n[0] = T{1};

        // Build degrees 1 .. p in place
        for (int j = 1; j <= deg; ++j) {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;

            T saved = T{0};

            for (int r = 0; r < j; ++r) {
                // Cox-de Boor coefficient for the two adjacent functions
                const T temp = n[r] / (right[r + 1] + left[j - r]);

                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            // Remaining contribution belongs to the last active function
            n[j] = saved;
        }
    }
}

template class BSpline<double>;

} // namespace iguana
