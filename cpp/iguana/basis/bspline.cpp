/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "bspline.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iguana
{

template<std::floating_point T>
BSpline<T>::BSpline(KnotVector<T> knots)
    : knots_(std::move(knots))
{
    if (degree() > max_degree)
        throw std::invalid_argument("BSpline: "
                                    "the degree exceeds max_degree");
}

template<std::floating_point T>
BSpline<T>::BSpline(int degree, std::vector<T> knots)
    : BSpline(KnotVector<T>(degree, std::move(knots)))
{
}

template<std::floating_point T>
void BSpline<T>::eval_on_element(int first_active, std::span<const T> points,
                                 Eigen::MatrixX<T>& values) const
{
    const int deg = degree();
    const Eigen::Index num_pts = static_cast<Eigen::Index>(points.size());
    const T* const knots = knots_.values().data();

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

template<std::floating_point T>
void BSpline<T>::derivs_on_element(
    int first_active, std::span<const T> points, int order,
    std::vector<Eigen::MatrixX<T>>& derivatives) const
{
    const int deg = degree();
    const Eigen::Index num_pts = static_cast<Eigen::Index>(points.size());
    const T* const knots = knots_.values().data();

    // Knot span shared by all evaluation points
    const int span = first_active + deg;

    // Orders above the degree vanish and are not computed
    const int computed = std::min(order, deg);

    derivatives.resize(static_cast<std::size_t>(order) + 1);

    for (Eigen::MatrixX<T>& derivative : derivatives)
        derivative.resize(deg + 1, num_pts);

    for (int k = computed + 1; k <= order; ++k)
        derivatives[k].setZero();

    // Functions of every degree above the diagonal, knot differences below
    T table[max_degree + 1][max_degree + 1];
    T left[max_degree + 1];
    T right[max_degree + 1];

    // Coefficients of the previous and current orders
    T coefficients[2][max_degree + 1];

    for (Eigen::Index q = 0; q < num_pts; ++q) {
        const T u = points[static_cast<std::size_t>(q)];

        table[0][0] = T{1};

        // Build degrees 1 .. p, keeping each of them
        for (int j = 1; j <= deg; ++j) {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;

            T saved = T{0};

            for (int r = 0; r < j; ++r) {
                table[j][r] = right[r + 1] + left[j - r];
                const T temp = table[r][j - 1] / table[j][r];

                table[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            table[j][j] = saved;
        }

        for (int r = 0; r <= deg; ++r)
            derivatives[0](r, q) = table[r][deg];

        // Order k combines the functions of degree p - k
        for (int r = 0; r <= deg; ++r) {
            int previous = 0;
            int current = 1;
            coefficients[previous][0] = T{1};

            for (int k = 1; k <= computed; ++k) {
                const int rk = r - k;
                const int pk = deg - k;
                T derivative = T{0};

                if (r >= k) {
                    coefficients[current][0] =
                        coefficients[previous][0] / table[pk + 1][rk];
                    derivative = coefficients[current][0] * table[rk][pk];
                }

                // Only functions on the span take part
                const int first = rk >= -1 ? 1 : -rk;
                const int last = r - 1 <= pk ? k - 1 : deg - r;

                for (int j = first; j <= last; ++j) {
                    coefficients[current][j] =
                        (coefficients[previous][j]
                         - coefficients[previous][j - 1])
                        / table[pk + 1][rk + j];
                    derivative +=
                        coefficients[current][j] * table[rk + j][pk];
                }

                if (r <= pk) {
                    coefficients[current][k] =
                        -coefficients[previous][k - 1] / table[pk + 1][r];
                    derivative += coefficients[current][k] * table[r][pk];
                }

                derivatives[k](r, q) = derivative;
                std::swap(previous, current);
            }
        }

        // Order k carries the factor p! / (p - k)!
        T factor = static_cast<T>(deg);

        for (int k = 1; k <= computed; ++k) {
            derivatives[k].col(q) *= factor;
            factor *= static_cast<T>(deg - k);
        }
    }
}

template class BSpline<double>;

} // namespace iguana
