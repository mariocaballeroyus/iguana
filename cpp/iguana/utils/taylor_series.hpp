/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_UTILS_TAYLOR_SERIES_HPP
#define IGUANA_UTILS_TAYLOR_SERIES_HPP

#include <algorithm>
#include <cmath>
#include <concepts>

#include <Eigen/Core>

namespace iguana
{

// Truncated Taylor series of a function about a point, by their
// coefficients c_k = f^(k) / k! there, from order 0 up to the last one kept

/**
 * @brief Coefficients of the product of two truncated Taylor series
 *
 * The product keeps the orders that both factors know
 *
 *     (a b)_n = sum_{k=0}^{n} a_k b_{n-k}
 *
 * @param a Coefficients of one factor
 * @param b Coefficients of the other factor
 * @return Coefficients of the product, as many as the shorter factor has
 */
template<std::floating_point T>
Eigen::VectorX<T> series_product(const Eigen::VectorX<T>& a,
                                 const Eigen::VectorX<T>& b)
{
    const Eigen::Index orders = std::min(a.size(), b.size());
    Eigen::VectorX<T> product = Eigen::VectorX<T>::Zero(orders);

    for (Eigen::Index n = 0; n < orders; ++n)
        for (Eigen::Index k = 0; k <= n; ++k)
            product(n) += a(k) * b(n - k);

    return product;
}

/**
 * @brief Coefficients of a real power of a truncated Taylor series
 *
 * Differentiating b = a^e gives a b' = e a' b, whose orders give each
 * coefficient from the lower ones
 *
 *     b_0 = a_0^e,
 *     b_n = sum_{k=1}^{n} (k (e + 1) - n) a_k b_{n-k} / (n a_0)
 *
 * @param a Coefficients of the series
 * @param exponent Real exponent e
 * @return Coefficients of the power, as many as @p a has
 *
 * @pre @p a is not empty and its first coefficient is positive
 */
template<std::floating_point T>
Eigen::VectorX<T> series_power(const Eigen::VectorX<T>& a, T exponent)
{
    Eigen::VectorX<T> power = Eigen::VectorX<T>::Zero(a.size());
    power(0) = std::pow(a(0), exponent);

    for (Eigen::Index n = 1; n < a.size(); ++n) {
        for (Eigen::Index k = 1; k <= n; ++k)
            power(n) += (static_cast<T>(k) * (exponent + 1)
                         - static_cast<T>(n)) * a(k) * power(n - k);

        power(n) /= static_cast<T>(n) * a(0);
    }

    return power;
}

/**
 * @brief Coefficients of the derivative of a truncated Taylor series
 *
 * Differentiating loses the highest order
 *
 *     (a')_k = (k + 1) a_{k+1}
 *
 * @param a Coefficients of the series
 * @return Coefficients of the derivative, one fewer than @p a has
 *
 * @pre @p a is not empty
 */
template<std::floating_point T>
Eigen::VectorX<T> series_derivative(const Eigen::VectorX<T>& a)
{
    Eigen::VectorX<T> derivative(a.size() - 1);

    for (Eigen::Index k = 0; k < derivative.size(); ++k)
        derivative(k) = static_cast<T>(k + 1) * a(k + 1);

    return derivative;
}

} // namespace iguana

#endif // IGUANA_UTILS_TAYLOR_SERIES_HPP
