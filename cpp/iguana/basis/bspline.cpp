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

    // Number of knots up to and including each distinct value
    for (std::size_t i = 0; i < knots_.size(); ++i) {
        if (i > 0 && knots_[i] == knots_[i - 1])
            // Repeated knot value: sum up to the last entry
            mult_sum_.back() = static_cast<int>(i) + 1;
        else
            // Distinct knot value: add new entry
            mult_sum_.push_back(static_cast<int>(i) + 1);
    }

    // Spans of the domain, from xi_p to xi_n
    const int last_span = num_functions() - 1;

    // The last distinct value has no span to its right
    const int last_entry = static_cast<int>(mult_sum_.size()) - 1;

    first_element_ = 0;
    num_elements_ = 0;

    for (int j = 0; j < last_entry; ++j) {
        // Span closed by the last appearance of the distinct value
        const int span = mult_sum_[static_cast<std::size_t>(j)] - 1;

        if (span < degree_)
            // Below the domain: the elements start after this value
            first_element_ = j + 1;
        else if (span <= last_span)
            // Inside the domain: a non-empty span, that is an element
            ++num_elements_;
    }
}

template<std::floating_point T>
void BSpline<T>::eval_on_element(int first_active, std::span<const T> points,
                                 Eigen::MatrixX<T>& values) const
{
    const int deg = degree_;
    const Eigen::Index num_pts = static_cast<Eigen::Index>(points.size());
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

template<std::floating_point T>
void BSpline<T>::eval_ders_on_element(int first_active,
                                      std::span<const T> points, int order,
                                      std::vector<Eigen::MatrixX<T>>& values)
    const
{
    const int deg = degree_;
    const Eigen::Index num_pts = static_cast<Eigen::Index>(points.size());
    const T* const knots = knots_.data();

    // Knot span to which the points belong
    const int span = first_active + deg;

    // One buffer per order
    values.resize(static_cast<std::size_t>(order) + 1);

    // Each buffer laid out as in eval_on_element()
    for (int k = 0; k <= order; ++k)
        values[static_cast<std::size_t>(k)].resize(deg + 1, num_pts);

    // Derivatives above the degree vanish
    for (int k = deg + 1; k <= order; ++k)
        values[static_cast<std::size_t>(k)].setZero();

    const int max_der = std::min(order, deg);

    // Recursion scratch, stack-allocated
    T ndu[max_degree + 1][max_degree + 1];
    T left[max_degree + 1];
    T right[max_degree + 1];

    // Two rows of coefficients, alternating between the derivative orders
    T coeff[2][max_degree + 1];

    for (Eigen::Index q = 0; q < num_pts; ++q) {
        // Parameter value of point `q`
        const T u = points[static_cast<std::size_t>(q)];

        // Evaluate p = 0, piecewise-constant
        ndu[0][0] = T{1};

        // Cox-de Boor recursion
        for (int j = 1; j <= deg; ++j) {
            // Left and right numerators, distances to knots
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;

            // Accumulate contribution
            T saved = T{0};

            for (int r = 0; r < j; ++r) {
                ndu[j][r] = right[r + 1] + left[j - r];

                // Cox-de Boor coefficients
                const T temp = ndu[r][j - 1] / ndu[j][r];

                // Update N_{i-j+1+r, j-1} -> N_{i-j+r, j}
                ndu[r][j] = saved + right[r + 1] * temp;

                // Carry remaining contribution to the next basis
                saved = left[j - r] * temp;
            }

            // Final basis function of degree j
            ndu[j][j] = saved;
        }

        // Order zero is the last column of the triangle
        T* const n = values[0].col(q).data();

        for (int j = 0; j <= deg; ++j) {
            n[j] = ndu[j][deg];
        }

        // Each function is differenced down the triangle in turn
        for (int r = 0; r <= deg; ++r) {
            // Coefficients of the previous and the current order
            T* prev = coeff[0];
            T* curr = coeff[1];

            // Order zero is the function itself
            prev[0] = T{1};

            for (int k = 1; k <= max_der; ++k) {
                // Row and degree of the triangle the order k reads from
                const int rk = r - k;
                const int pk = deg - k;

                // Accumulate the derivative of order k
                T d = T{0};

                // Leading coefficient, absent for the lowest functions
                if (r >= k) {
                    curr[0] = prev[0] / ndu[pk + 1][rk];
                    d = curr[0] * ndu[rk][pk];
                }

                // The range is clipped where the triangle has no entry
                const int j1 = (rk >= -1) ? 1 : -rk;
                const int j2 = (r - 1 <= pk) ? k - 1 : deg - r;

                // Interior coefficients, differences of the previous order
                for (int j = j1; j <= j2; ++j) {
                    curr[j] = (prev[j] - prev[j - 1]) / ndu[pk + 1][rk + j];
                    d += curr[j] * ndu[rk + j][pk];
                }

                // Trailing coefficient, absent for the highest functions
                if (r <= pk) {
                    curr[k] = -prev[k - 1] / ndu[pk + 1][r];
                    d += curr[k] * ndu[r][pk];
                }

                values[static_cast<std::size_t>(k)](r, q) = d;

                // The current order becomes the previous one
                std::swap(prev, curr);
            }
        }

        // The differences carry the falling factorial p!/(p-k)!
        int factor = deg;

        for (int k = 1; k <= max_der; ++k) {
            T* const dk =
                values[static_cast<std::size_t>(k)].col(q).data();

            for (int j = 0; j <= deg; ++j) {
                dk[j] *= static_cast<T>(factor);
            }

            factor *= (deg - k);
        }
    }
}

template class BSpline<double>;

} // namespace iguana
