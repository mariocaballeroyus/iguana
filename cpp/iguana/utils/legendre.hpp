/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_UTILS_LEGENDRE_HPP
#define IGUANA_UTILS_LEGENDRE_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

#include <Eigen/Core>

#include "iguana/utils/multi_index.hpp"

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

/**
 * @brief Evaluates the tensor-product Legendre polynomials up to a degree
 *        in each direction
 *
 * The products P_{j_1}(xi_1) ... P_{j_d}(xi_d), with every j_k from 0 to
 * degree, are ordered with j_1 running fastest, as the multi-indices of
 * the library run
 *
 * @param degree Highest degree of each direction
 * @param points Points at which the products are evaluated, usually in
 *        [-1, 1]^d, with size (num_points, d)
 * @param values Output matrix with size ((degree + 1)^d, num_points), one
 *        row per product. It is resized when necessary
 *
 * @pre @p degree is non-negative and @p points has d columns
 */
template<std::floating_point T, std::size_t d>
void tensor_legendre_polynomials(int degree,
                                 const Eigen::MatrixX<T>& points,
                                 Eigen::MatrixX<T>& values)
{
    const Eigen::Index num_points = points.rows();

    // Polynomials of each direction, one row per degree
    std::array<Eigen::MatrixX<T>, d> univariate;

    for (std::size_t direction = 0; direction < d; ++direction) {
        const std::span<const T> coordinates(points.col(direction).data(),
                                             num_points);

        legendre_polynomials(degree, coordinates, univariate[direction]);
    }

    std::array<int, d> degrees{};
    std::array<int, d> bounds{};
    bounds.fill(degree + 1);

    int num_products = 1;

    for (std::size_t direction = 0; direction < d; ++direction)
        num_products *= degree + 1;

    values.resize(num_products, num_points);

    // One row per product, with the first degree running fastest
    int product = 0;

    do {
        values.row(product) = univariate[0].row(degrees[0]);

        for (std::size_t direction = 1; direction < d; ++direction)
            values.row(product).array() *=
                univariate[direction].row(degrees[direction]).array();

        ++product;
    } while (next_lexicographic(degrees, bounds));
}

} // namespace iguana

#endif // IGUANA_UTILS_LEGENDRE_HPP
