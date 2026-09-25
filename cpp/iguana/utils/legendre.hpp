/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_UTILS_LEGENDRE_HPP
#define IGUANA_UTILS_LEGENDRE_HPP

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Evaluates the Legendre polynomials up to a degree
 *
 * The polynomials P_0 to P_degree are orthogonal on [-1, 1], with norms
 * 2 / (2n + 1), and follow the three-term recurrence
 *
 *     (n + 1) P_{n+1}(xi) = (2n + 1) xi P_n(xi) - n P_{n-1}(xi)
 *
 * @param degree Highest degree
 * @param points Points at which the polynomials are evaluated, usually in
 *        [-1, 1]
 * @param values Output matrix with size (degree + 1, num_points), one row
 *        per polynomial as BSpline::eval_on_element() lays them out. It is
 *        resized when necessary
 *
 * @pre @p degree is non-negative
 */
template<std::floating_point T>
void legendre_polynomials(int degree,
                          std::type_identity_t<std::span<const T>> points,
                          Eigen::MatrixX<T>& values)
{
    values.resize(degree + 1, points.size());

    for (std::size_t point = 0; point < points.size(); ++point) {
        const T xi = points[point];

        values(0, point) = T{1};

        if (degree > 0)
            values(1, point) = xi;

        // (n + 1) P_{n+1} = (2n + 1) xi P_n - n P_{n-1}
        for (int n = 1; n < degree; ++n)
            values(n + 1, point) = ((2 * n + 1) * xi * values(n, point)
                                    - n * values(n - 1, point))
                                   / (n + 1);
    }
}

/**
 * @brief Evaluates the antiderivatives of the Legendre polynomials up to a
 *        degree that vanish at -1
 *
 * Each antiderivative follows from the polynomials one degree above and
 * below,
 *
 *     int_{-1}^{xi} P_n = (P_{n+1}(xi) - P_{n-1}(xi)) / (2n + 1)
 *
 * and is xi + 1 for P_0
 *
 * @param degree Highest degree
 * @param points Points at which the antiderivatives are evaluated, usually
 *        in [-1, 1]
 * @param values Output matrix with size (degree + 1, num_points), laid out
 *        as in legendre_polynomials(). It is resized when necessary
 *
 * @pre @p degree is non-negative
 */
template<std::floating_point T>
void legendre_antiderivatives(int degree,
                              std::type_identity_t<std::span<const T>> points,
                              Eigen::MatrixX<T>& values)
{
    values.resize(degree + 1, points.size());

    for (std::size_t point = 0; point < points.size(); ++point) {
        const T xi = points[point];

        // P_{n-1} and P_n, advanced by the recurrence one degree at a time
        T previous{1};
        T current = xi;

        // xi + 1 = P_1 + 1 for P_0
        values(0, point) = current + T{1};

        for (int n = 1; n <= degree; ++n) {
            const T next = ((2 * n + 1) * xi * current - n * previous)
                           / (n + 1);

            values(n, point) = (next - previous) / (2 * n + 1);
            previous = current;
            current = next;
        }
    }
}

} // namespace iguana

#endif // IGUANA_UTILS_LEGENDRE_HPP
