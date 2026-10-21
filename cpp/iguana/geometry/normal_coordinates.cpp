/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "normal_coordinates.hpp"

#include <algorithm>
#include <stdexcept>

#include "iguana/utils/multi_index.hpp"
#include "iguana/utils/taylor_series.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Jet of a derivative operator composed after the derivative along a
 *        direction that varies along the curve
 *
 *     (e · ∇) ∘ Σ_α c_α ∂^α = Σ_α (e_x c_α ∂^(α + (1, 0))
 *                                  + e_y c_α ∂^(α + (0, 1)))
 *
 * @details The jets hold one row per Taylor coefficient along the curve.
 *          The operator has one column per multi-index of total order at
 *          most @p order, and none of the highest order, which would be
 *          raised past it
 *
 * @param direction Jet of the direction, its two components as columns
 * @param op Jet of the coefficients of the operator
 * @param order Highest total order of the multi-indices
 * @return Jet of the composed operator, as long as the shorter factor
 */
template<std::floating_point T>
Eigen::MatrixX<T> compose(const Eigen::MatrixX<T>& direction,
                          const Eigen::MatrixX<T>& op, int order)
{
    const Eigen::MatrixX<T> along_x =
        series_product<T>(direction.col(0), op);
    const Eigen::MatrixX<T> along_y =
        series_product<T>(direction.col(1), op);
    Eigen::MatrixX<T> composed =
        Eigen::MatrixX<T>::Zero(along_x.rows(), op.cols());

    for (int b = 0; b < order; ++b) {
        for (int a = 0; a + b < order; ++a) {
            const int from = simplex_index(a, b, order);

            composed.col(simplex_index(a + 1, b, order)) += along_x.col(from);
            composed.col(simplex_index(a, b + 1, order)) += along_y.col(from);
        }
    }

    return composed;
}

} // namespace

template<std::floating_point T>
Eigen::MatrixX<T> normal_coordinate_derivs(const Eigen::Matrix2<T>& frame,
                                           const Eigen::VectorX<T>& curvatures,
                                           int order)
{
    if (order < 0)
        throw std::invalid_argument("normal_coordinate_derivs: "
                                    "the order must be non-negative");

    if (curvatures.size() < order - 1)
        throw std::invalid_argument("normal_coordinate_derivs: "
                                    "there must be a curvature derivative "
                                    "below each order but the last");

    const int size = simplex_size(order);
    Eigen::MatrixX<T> derivs = Eigen::MatrixX<T>::Zero(size, size);

    // Jets along the curve, with Taylor coefficients κ^(l) / l!
    const int length = std::max(order, 1);
    Eigen::VectorX<T> kappa(length - 1);
    T factorial{1};

    for (int l = 0; l < length - 1; ++l) {
        if (l > 0)
            factorial *= static_cast<T>(l);

        kappa(l) = curvatures(l) / factorial;
    }

    // τ' = −κ ν and ν' = κ τ, one order at a time
    Eigen::MatrixX<T> tangent(length, 2);
    Eigen::MatrixX<T> normal(length, 2);
    tangent.row(0) = frame.col(0).transpose();
    normal.row(0) = frame.col(1).transpose();

    for (int l = 0; l + 1 < length; ++l) {
        Eigen::RowVector2<T> kappa_normal = Eigen::RowVector2<T>::Zero();
        Eigen::RowVector2<T> kappa_tangent = Eigen::RowVector2<T>::Zero();

        for (int k = 0; k <= l; ++k) {
            kappa_normal += kappa(k) * normal.row(l - k);
            kappa_tangent += kappa(k) * tangent.row(l - k);
        }

        tangent.row(l + 1) = -kappa_normal / static_cast<T>(l + 1);
        normal.row(l + 1) = kappa_tangent / static_cast<T>(l + 1);
    }

    for (int i = 0; i <= order; ++i) {
        // (ν · ∇)^i, as long as its derivatives along the curve need
        Eigen::MatrixX<T> op = Eigen::MatrixX<T>::Zero(order - i + 1, size);
        op(0, simplex_index(0, 0, order)) = T{1};

        for (int k = 0; k < i; ++k)
            op = compose<T>(normal, op, order).topRows(order - i + 1);

        // d/ds (Σ_α c_α ∂^α) = Σ_α c_α' ∂^α + (τ · ∇) ∘ Σ_α c_α ∂^α
        for (int j = 0; j + i <= order; ++j) {
            derivs.row(simplex_index(j, i, order)) = op.row(0);

            if (j + i < order) {
                const Eigen::Index rows = op.rows() - 1;
                op = series_derivative(op)
                     + compose<T>(tangent, op, order).topRows(rows);
            }
        }
    }

    return derivs;
}

template Eigen::MatrixX<double> normal_coordinate_derivs(
    const Eigen::Matrix2<double>&, const Eigen::VectorX<double>&, int);

} // namespace iguana
