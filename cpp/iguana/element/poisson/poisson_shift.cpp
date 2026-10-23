/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "poisson_shift.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/geometry/directional_derivs.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<typename Basis> requires (Basis::dimension == 2)
PoissonShift<typename Basis::Scalar> poisson_shift(
    const Patch<Basis, 2>& patch,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order)
{
    using T = typename Basis::Scalar;

    if (order < 0)
        throw std::invalid_argument("poisson_shift: "
                                    "the order must be non-negative");

    if (!patch.is_affine())
        throw std::invalid_argument("poisson_shift: "
                                    "the map of the patch must be affine");

    const int m = order;
    const Eigen::Index num_points = projection.positions.rows();

    PoissonShift<T> shift{
        Eigen::MatrixX<T>::Zero(num_points, simplex_size(m)),
        Eigen::MatrixX<T>::Zero(num_points, m >= 2 ? simplex_size(m - 2)
                                                   : 0)};

    const Eigen::Matrix2<T> linear = patch.linear_part();
    const Eigen::Matrix2<T> inverse = linear.inverse();

    for (Eigen::Index point = 0; point < num_points; ++point) {
        // The physical vector from the closest point to the point, of
        // length t along e
        const Eigen::Vector2<T> offset =
            -linear * projection.distances.row(point).transpose();
        const T t = offset.norm();

        // At the closest point itself every term but the value vanishes
        const T scale = std::max(T{1}, projection.positions.row(point)
                                           .cwiseAbs().maxCoeff());

        if (t <= 64 * std::numeric_limits<T>::epsilon() * scale)
            continue;

        // The directions of ∂_⊥ and ∂_e, mapped into the parameters for the
        // field, and Cartesian for the source
        const Eigen::Vector2<T> along = offset / t;
        const Eigen::Vector2<T> across(-along(1), along(0));

        Eigen::Matrix2<T> frame;
        frame << across, along;

        const Eigen::MatrixX<T> parametric =
            directional_derivs<T>(inverse * frame, m);
        const Eigen::MatrixX<T> cartesian =
            directional_derivs<T>(frame, std::max(m - 2, 0));

        T power{1};

        for (int k = 1; k <= m; ++k) {
            power *= t / static_cast<T>(k);

            // The term of order k of the series with those of the residual,
            // which leave at most one derivative along e
            const int j = k / 2;
            const T sign = j % 2 == 0 ? T{1} : T{-1};

            shift.weights.row(point) -=
                sign * power
                * parametric.row(simplex_index(2 * j, k - 2 * j, m));

            // The source of the terms of the residual, D_k f
            for (int i = 0; 2 * i + 2 <= k; ++i) {
                const T term = i % 2 == 0 ? power : -power;

                shift.source_weights.row(point) -=
                    term * cartesian.row(simplex_index(2 * i, k - 2 - 2 * i,
                                                       m - 2));
            }
        }
    }

    return shift;
}

template PoissonShift<double> poisson_shift(
    const Patch<TensorBSpline<double, 2>, 2>&,
    const BoundaryProjection<double>&, int);

} // namespace iguana
