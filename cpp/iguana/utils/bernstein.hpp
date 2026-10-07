/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_UTILS_BERNSTEIN_HPP
#define IGUANA_UTILS_BERNSTEIN_HPP

#include <algorithm>
#include <concepts>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Value of a polynomial in Bernstein form on [0, 1] at a point, by
 *        the de Casteljau algorithm
 *
 * Each level replaces neighboring coefficients by their convex combination
 * (1 - t) b_j + t b_{j + 1}, which keeps the evaluation stable
 *
 * @param coefficients Bernstein coefficients b_0, ..., b_p
 * @param t Point of [0, 1]
 * @return Value of the polynomial at @p t
 *
 * @pre @p coefficients is not empty
 */
template<std::floating_point T>
T de_casteljau(const Eigen::VectorX<T>& coefficients, T t)
{
    Eigen::VectorX<T> work = coefficients;

    for (Eigen::Index level = work.size() - 1; level > 0; --level)
        for (Eigen::Index j = 0; j < level; ++j)
            work(j) = (1 - t) * work(j) + t * work(j + 1);

    return work(0);
}

/**
 * @brief Bernstein coefficients of the two parts of a polynomial, on [0, t]
 *        and on [t, 1], each rescaled to [0, 1]
 *
 * The left part takes the first coefficient of each level of the de
 * Casteljau algorithm and the right part the last one, so that both share
 * the value at @p t
 *
 * @param coefficients Bernstein coefficients on [0, 1]
 * @param t Point of [0, 1] at which the polynomial is split
 * @param left Output coefficients of the part on [0, t], resized when
 *        necessary
 * @param right Output coefficients of the part on [t, 1], resized when
 *        necessary
 *
 * @pre @p coefficients is not empty
 */
template<std::floating_point T>
void de_casteljau_split(const Eigen::VectorX<T>& coefficients, T t,
                        Eigen::VectorX<T>& left, Eigen::VectorX<T>& right)
{
    const Eigen::Index size = coefficients.size();
    Eigen::VectorX<T> work = coefficients;

    left.resize(size);
    right.resize(size);

    for (Eigen::Index level = 0; level < size; ++level) {
        const Eigen::Index last = size - 1 - level;

        left(level) = work(0);
        right(last) = work(last);

        for (Eigen::Index j = 0; j < last; ++j)
            work(j) = (1 - t) * work(j) + t * work(j + 1);
    }
}

/**
 * @brief Bernstein coefficients of the derivative of a polynomial in
 *        Bernstein form on [0, 1]
 *
 * The derivative of a polynomial of degree p has degree p - 1, with
 * coefficients p (b_{j + 1} - b_j). That of a constant is zero, given as a
 * single zero coefficient
 *
 * @param coefficients Bernstein coefficients b_0, ..., b_p
 * @return Bernstein coefficients of the derivative
 *
 * @pre @p coefficients is not empty
 */
template<std::floating_point T>
Eigen::VectorX<T> bernstein_derivative(const Eigen::VectorX<T>& coefficients)
{
    const Eigen::Index degree = coefficients.size() - 1;

    if (degree == 0)
        return Eigen::VectorX<T>::Zero(1);

    return static_cast<T>(degree)
           * (coefficients.tail(degree) - coefficients.head(degree));
}

/**
 * @brief Bernstein coefficients of the product of two polynomials in
 *        Bernstein form on [0, 1]
 *
 * The product of degrees m and n has degree m + n, with coefficients
 * c_k = sum over i + j = k of C(m, i) C(n, j) / C(m + n, k) a_i b_j
 *
 * @param left Bernstein coefficients a_0, ..., a_m
 * @param right Bernstein coefficients b_0, ..., b_n
 * @return Bernstein coefficients of the product
 *
 * @pre Neither @p left nor @p right is empty
 */
template<std::floating_point T>
Eigen::VectorX<T> bernstein_product(const Eigen::VectorX<T>& left,
                                    const Eigen::VectorX<T>& right)
{
    // Binomial coefficients C(degree, k) for k = 0, ..., degree
    const auto binomials = [](Eigen::Index degree) {
        Eigen::VectorX<T> result(degree + 1);
        result(0) = T{1};

        for (Eigen::Index k = 1; k <= degree; ++k)
            result(k) = result(k - 1) * static_cast<T>(degree - k + 1)
                        / static_cast<T>(k);

        return result;
    };

    const Eigen::Index left_degree = left.size() - 1;
    const Eigen::Index right_degree = right.size() - 1;
    const Eigen::VectorX<T> left_binomials = binomials(left_degree);
    const Eigen::VectorX<T> right_binomials = binomials(right_degree);

    Eigen::VectorX<T> product =
        Eigen::VectorX<T>::Zero(left_degree + right_degree + 1);

    for (Eigen::Index i = 0; i <= left_degree; ++i)
        for (Eigen::Index j = 0; j <= right_degree; ++j)
            product(i + j) += left_binomials(i) * right_binomials(j)
                              * left(i) * right(j);

    return product.cwiseQuotient(binomials(left_degree + right_degree));
}

/**
 * @brief Number of sign changes of some Bernstein coefficients, zeros
 *        skipped
 *
 * By Descartes' rule of signs in Bernstein form, it bounds the number of
 * roots of the polynomial in (0, 1), counted with multiplicity, and has
 * the same parity
 *
 * @param coefficients Bernstein coefficients
 * @return Number of sign changes
 */
template<std::floating_point T>
int sign_changes(const Eigen::VectorX<T>& coefficients)
{
    int changes = 0;
    T previous{0};

    for (const T coefficient : coefficients) {
        if (coefficient == T{0})
            continue;

        if (previous != T{0} && (coefficient > T{0}) != (previous > T{0}))
            ++changes;

        previous = coefficient;
    }

    return changes;
}

/**
 * @brief Points of (0, 1) at which a polynomial in Bernstein form changes
 *        sign, in increasing order
 *
 * Intervals are split in halves until each holds at most one sign change
 * of its coefficients, and one with a single change holds a single root,
 * which bisection refines. A zero exactly at a split point is a crossing
 * where the polynomial changes sign there. Below the tolerance width an
 * interval is not split further, and its middle is a crossing if it holds
 * an odd number of sign changes, so that roots of even multiplicity, which
 * only touch zero, are left out. Zeros at 0 and 1 are never reported
 *
 * @param coefficients Bernstein coefficients on [0, 1]
 * @param tolerance Width below which intervals are not split further
 * @return Crossings in increasing order
 *
 * @pre @p coefficients is not empty and @p tolerance is positive
 */
template<std::floating_point T>
std::vector<T> bernstein_crossings(const Eigen::VectorX<T>& coefficients,
                                   T tolerance)
{
    // Sign of the polynomial just inside the start or the end of an
    // interval, that of its first or last nonzero coefficient
    const auto start_sign = [](const Eigen::VectorX<T>& part) {
        for (const T coefficient : part)
            if (coefficient != T{0})
                return coefficient > T{0} ? 1 : -1;

        return 0;
    };

    const auto end_sign = [](const Eigen::VectorX<T>& part) {
        for (Eigen::Index j = part.size() - 1; j >= 0; --j)
            if (part(j) != T{0})
                return part(j) > T{0} ? 1 : -1;

        return 0;
    };

    // Part of [0, 1] still to examine, with the coefficients of the
    // polynomial on it
    struct Interval
    {
        T start;
        T end;
        Eigen::VectorX<T> coefficients;
    };

    std::vector<T> crossings;
    std::vector<Interval> pending{{T{0}, T{1}, coefficients}};
    Eigen::VectorX<T> left;
    Eigen::VectorX<T> right;

    while (!pending.empty()) {
        const Interval interval = std::move(pending.back());
        pending.pop_back();

        const int changes = sign_changes(interval.coefficients);
        const T middle = (interval.start + interval.end) / 2;

        if (changes == 0)
            continue;

        if (changes == 1) {
            // Bisect the polynomial between the ends of the interval, which
            // keep their signs just inside it
            T low = interval.start;
            T high = interval.end;
            const int low_sign = start_sign(interval.coefficients);

            for (int step = 0; step < 64; ++step) {
                const T point = (low + high) / 2;

                if (point <= low || point >= high)
                    break;

                const T value = de_casteljau(coefficients, point);

                if (value == T{0}) {
                    low = high = point;
                    break;
                }

                if ((value > T{0}) == (low_sign > 0))
                    low = point;
                else
                    high = point;
            }

            crossings.push_back((low + high) / 2);
            continue;
        }

        if (interval.end - interval.start < tolerance) {
            if (changes % 2 == 1)
                crossings.push_back(middle);

            continue;
        }

        de_casteljau_split(interval.coefficients, T{0.5}, left, right);

        // A zero at the split point is a crossing where the sign changes
        if (right(0) == T{0} && end_sign(left) * start_sign(right) < 0)
            crossings.push_back(middle);

        pending.push_back({middle, interval.end, right});
        pending.push_back({interval.start, middle, left});
    }

    std::ranges::sort(crossings);

    return crossings;
}

} // namespace iguana

#endif // IGUANA_UTILS_BERNSTEIN_HPP
