/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "bspline.hpp"

#include<algorithm>
#include<cstddef>
#include<stdexcept>
#include<utility>

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

    const std::size_t min_knots = 2 * (static_cast<std::size_t>(degree_) + 1);

    if (knots_.size() < min_knots)
        throw std::invalid_argument("BSpline: "
                                    "too few knots for the given degree");
}

template<std::floating_point T>
void BSpline<T>::eval_on_element(int first_active, std::span<const T> points,
                                 Eigen::MatrixX<T>& values) const
{
    const int deg = degree_;
    const auto num_pts = static_cast<Eigen::Index>(points.size());
    const T* const knots = knots_.data();

    // Knot span to which the points belong
    const int span = first_active + deg;

    // Values buffer, reused across elements
    values.resize(deg + 1, num_pts);

    // Recursion scratch, stack-allocated
    T left[max_degree + 1];
    T right[max_degree + 1];

    for (Eigen::Index q = 0; q < num_pts; ++q) {
        // Parameter value of point `q`
        const T u = points[static_cast<std::size_t>(q)];
        // Basis values at point `q`, recursion built in-place
        T* const n = values.col(q).data();

        // Evaluate p = 0, piecewise-constant
        n[0] = T{1};

        // Evaluate p = 1 .. deg, Cox-de Boor recursion
        for (int j = 1; j <= deg; ++j) {
            // Left and right numerators, distances to knots
            left[j]  = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;

            // Accumulate contribution
            T saved = T{0};

            for (int r = 0; r < j; ++r) {
                // Cox-de Boor coefficients
                const T temp = n[r] / (right[r + 1] + left[j - r]);

                // Update N_{i-j+1+r, j-1} -> N_{i-j+r, j}
                n[r] = saved + right[r + 1] * temp;

                // Carry remaining contribution to the next basis
                saved = left[j - r] * temp;
            }

            // Final basis function of degree j
            n[j] = saved;
        }
    }
}

template class BSpline<double>;

} // namespace iguana
