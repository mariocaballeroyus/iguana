/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "poisson_normal_series.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "iguana/utils/multi_index.hpp"
#include "iguana/utils/taylor_series.hpp"

namespace iguana
{

template<std::floating_point T>
PoissonNormalSeries<T> poisson_normal_series(
    const Eigen::VectorX<T>& curvatures, int order)
{
    if (order < 0)
        throw std::invalid_argument("poisson_normal_series: "
                                    "the order must be non-negative");

    if (curvatures.size() < order - 1)
        throw std::invalid_argument("poisson_normal_series: "
                                    "there must be a curvature derivative "
                                    "below each order but the last");

    const int m = order;
    const int field_size = simplex_size(m);
    const int source_size = m >= 2 ? simplex_size(m - 2) : 0;

    // Each jet along the boundary holds the Taylor coefficients of a form,
    // its derivatives over l!, over the derivatives of u and then of f.
    // U_k needs those up to r(k) = 2 ⌊(m − k) / 2⌋ for the higher orders
    const auto length = [m](int k) { return 2 * ((m - k) / 2) + 1; };
    const auto seed = [&](int count, auto column_of_l) {
        Eigen::MatrixX<T> jet =
            Eigen::MatrixX<T>::Zero(count, field_size + source_size);
        T factorial{1};

        for (int l = 0; l < count; ++l) {
            if (l > 0)
                factorial *= static_cast<T>(l);

            jet(l, column_of_l(l)) = T{1} / factorial;
        }

        return jet;
    };

    // U_0 = g and U_1 = q, and the derivatives F_i of f along the normal
    std::vector<Eigen::MatrixX<T>> U(static_cast<std::size_t>(m) + 1);
    std::vector<Eigen::MatrixX<T>> F;

    U[0] = seed(length(0), [m](int l) { return simplex_index(l, 0, m); });

    if (m >= 1)
        U[1] = seed(length(1),
                    [m](int l) { return simplex_index(l, 1, m); });

    for (int i = 0; i + 2 <= m; ++i)
        F.push_back(seed(length(i + 2), [&](int l) {
            return field_size + simplex_index(l, i, m - 2);
        }));

    // κ and its powers as series, κ^(l) / l!
    Eigen::VectorX<T> kappa(std::max(m - 1, 1));
    kappa.setZero();
    T factorial{1};

    for (int l = 0; l < m - 1; ++l) {
        if (l > 0)
            factorial *= static_cast<T>(l);

        kappa(l) = curvatures(l) / factorial;
    }

    std::vector<Eigen::VectorX<T>> powers{
        Eigen::VectorX<T>::Unit(kappa.size(), 0)};

    for (int j = 1; j <= 3; ++j)
        powers.push_back(series_product(kappa, powers.back()));

    const Eigen::VectorX<T> kappa_prime =
        kappa.size() > 1 ? series_derivative(kappa)
                         : Eigen::VectorX<T>::Zero(1);

    // ∂_t^n at t = 0 of the equation, with C(3, j), C(2, j) and n! / (n − j)!
    //
    //     U_{n+2} = − Σ_j C(3, j) n!/(n−j)! κ^j F_{n−j}
    //               − Σ_{j≥1} C(3, j) n!/(n−j)! κ^j U_{n−j+2}
    //               − Σ_j C(2, j) n!/(n−j)! κ^(j+1) U_{n−j+1}
    //               − U_n'' − n κ U_{n−1}'' + n κ' U_{n−1}'
    const std::array<T, 4> binomial3{T{1}, T{3}, T{3}, T{1}};
    const std::array<T, 3> binomial2{T{1}, T{2}, T{1}};

    for (int n = 0; n + 2 <= m; ++n) {
        const Eigen::Index rows = length(n + 2);
        const auto times = [rows](const Eigen::VectorX<T>& series,
                                  const Eigen::MatrixX<T>& jet) {
            return Eigen::MatrixX<T>(
                series_product(series, jet).topRows(rows));
        };

        Eigen::MatrixX<T> next = -Eigen::MatrixX<T>(
            series_derivative(series_derivative(
                U[static_cast<std::size_t>(n)])).topRows(rows));
        T falling{1};

        for (int j = 0; j <= std::min(3, n); ++j) {
            if (j > 0)
                falling *= static_cast<T>(n - j + 1);

            const auto ju = static_cast<std::size_t>(j);
            const T factor = binomial3[ju] * falling;

            next -= factor
                    * times(powers[ju], F[static_cast<std::size_t>(n - j)]);

            if (j >= 1)
                next -= factor
                        * times(powers[ju],
                                U[static_cast<std::size_t>(n - j + 2)]);

            if (j <= 2)
                next -= binomial2[ju] * falling
                        * times(powers[ju + 1],
                                U[static_cast<std::size_t>(n - j + 1)]);
        }

        if (n >= 1) {
            const Eigen::MatrixX<T>& lower = U[static_cast<std::size_t>(n - 1)];
            const Eigen::MatrixX<T> first = series_derivative(lower);

            next -= static_cast<T>(n)
                    * times(kappa, series_derivative(first));
            next += static_cast<T>(n) * times(kappa_prime, first);
        }

        U[static_cast<std::size_t>(n) + 2] = next;
    }

    // The forms themselves, the first coefficients of the jets
    PoissonNormalSeries<T> series{
        Eigen::MatrixX<T>(m + 1, field_size),
        Eigen::MatrixX<T>(m + 1, source_size)};

    for (int k = 0; k <= m; ++k) {
        const Eigen::MatrixX<T>& jet = U[static_cast<std::size_t>(k)];

        series.field.row(k) = jet.row(0).head(field_size);
        series.source.row(k) = jet.row(0).tail(source_size);
    }

    return series;
}

template PoissonNormalSeries<double> poisson_normal_series(
    const Eigen::VectorX<double>&, int);

} // namespace iguana
