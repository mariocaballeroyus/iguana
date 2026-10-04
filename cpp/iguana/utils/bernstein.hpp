/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_UTILS_BERNSTEIN_HPP
#define IGUANA_UTILS_BERNSTEIN_HPP

#include <concepts>

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

} // namespace iguana

#endif // IGUANA_UTILS_BERNSTEIN_HPP
