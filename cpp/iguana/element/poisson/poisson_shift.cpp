/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "poisson_shift.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/element/poisson/poisson_normal_series.hpp"
#include "iguana/geometry/normal_coordinates.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<typename Basis> requires (Basis::dimension == 2)
PoissonShift<typename Basis::Scalar> poisson_shift(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order)
{
    using T = typename Basis::Scalar;

    if (order < 0)
        throw std::invalid_argument("poisson_shift: "
                                    "the order must be non-negative");

    if (!patch.is_affine())
        throw std::invalid_argument("poisson_shift: "
                                    "the map of the patch must be affine");

    // The field enters through ∂_s^j ∂_t u with j up to 2 ⌊(m − 1) / 2⌋
    const int m = order;
    const int r = m >= 1 ? 2 * ((m - 1) / 2) + 1 : 0;
    const Eigen::Index num_points = projection.positions.rows();

    PoissonShift<T> shift{
        r, Eigen::MatrixX<T>::Zero(num_points, simplex_size(r)),
        Eigen::MatrixX<T>::Zero(num_points, simplex_size(m)),
        Eigen::MatrixX<T>::Zero(num_points, m >= 2 ? simplex_size(m - 2)
                                                   : 0)};

    const Eigen::MatrixX<T> curvatures =
        curvature_derivs(boundary, projection, std::max(m - 2, 0));
    const Eigen::Matrix2<T> linear = patch.linear_part();
    const Eigen::Matrix2<T> inverse = linear.inverse();

    for (Eigen::Index point = 0; point < num_points; ++point) {
        // The outward normal ν, the tangent τ turned from it, and the
        // physical vector from the closest point to the point
        const Eigen::Vector2<T> normal = projection.normals.row(point);
        const Eigen::Vector2<T> tangent(-normal(1), normal(0));
        const Eigen::Vector2<T> offset =
            -linear * projection.distances.row(point).transpose();

        const T scale = std::max(T{1}, projection.positions.row(point)
                                           .cwiseAbs().maxCoeff());
        const T floor = 64 * std::numeric_limits<T>::epsilon() * scale;

        if (std::abs(offset.dot(tangent)) > T{1e-6} * offset.norm() + floor)
            throw std::invalid_argument("poisson_shift: "
                                        "the distance vectors must lie along "
                                        "the normals of a smooth boundary");

        // L = Σ_k t^k / k! U_k, over the derivatives of u and of f
        const Eigen::VectorX<T> curvature = curvatures.row(point);
        const PoissonNormalSeries<T> series =
            poisson_normal_series(curvature, m);
        const T t = offset.dot(normal);

        Eigen::RowVectorX<T> field =
            Eigen::RowVectorX<T>::Zero(series.field.cols());
        Eigen::RowVectorX<T> source =
            Eigen::RowVectorX<T>::Zero(series.source.cols());
        T power{1};

        for (int k = 0; k <= m; ++k) {
            if (k > 0)
                power *= t / static_cast<T>(k);

            field += power * series.field.row(k);
            source += power * series.source.row(k);
        }

        // The normal derivative and its derivatives along the boundary from
        // the parametric ones of the field, through the frame mapped into
        // the parameters, the trace and the source from Cartesian ones
        Eigen::Matrix2<T> frame;
        frame << tangent, normal;

        const Eigen::MatrixX<T> parametric =
            normal_coordinate_derivs<T>(inverse * frame, curvature, m);
        const Eigen::MatrixX<T> cartesian =
            normal_coordinate_derivs<T>(frame, curvature, m);

        for (int j = 0; j <= m; ++j) {
            const int along = simplex_index(j, 0, m);
            shift.value_weights.row(point) +=
                field(along) * cartesian.row(along);

            if (j + 1 > m)
                continue;

            // The parametric multi-indices of total order up to r
            const int across = simplex_index(j, 1, m);

            for (int b = 0; b <= r; ++b)
                for (int a = 0; a + b <= r; ++a)
                    shift.weights(point, simplex_index(a, b, r)) -=
                        field(across) * parametric(across,
                                                   simplex_index(a, b, m));
        }

        if (m >= 2) {
            shift.source_weights.row(point) =
                source * normal_coordinate_derivs<T>(frame, curvature, m - 2);
        }
    }

    return shift;
}

template PoissonShift<double> poisson_shift(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    const BoundaryProjection<double>&, int);
template PoissonShift<double> poisson_shift(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    const BoundaryProjection<double>&, int);

} // namespace iguana
