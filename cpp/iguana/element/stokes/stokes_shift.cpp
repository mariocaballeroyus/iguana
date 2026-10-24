/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "stokes_shift.hpp"

#include <stdexcept>

#include <Eigen/LU>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/element/poisson/poisson_shift.hpp"
#include "iguana/geometry/directional_derivs.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<typename Basis> requires (Basis::dimension == 2)
StokesShift<typename Basis::Scalar> stokes_shift(
    const Patch<Basis, 2>& patch,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order,
    typename Basis::Scalar viscosity)
{
    using T = typename Basis::Scalar;

    if (viscosity <= 0)
        throw std::invalid_argument("stokes_shift: "
                                    "the viscosity must be positive");

    const PoissonShift<T> base = poisson_shift(patch, projection, order);
    const int m = order;
    const Eigen::Index num_points = base.weights.rows();

    StokesShift<T> shift{
        base.weights,
        std::vector<Eigen::MatrixX<T>>(
            2, Eigen::MatrixX<T>::Zero(num_points, simplex_size(m))),
        base.source_weights / viscosity};

    if (m < 2)
        return shift;

    // The Cartesian derivatives of the pressure gradient, up to order
    // m − 1, as parametric ones of the affine map
    const Eigen::MatrixX<T> cartesian =
        directional_derivs<T>(patch.linear_part().inverse(), m - 1);

    // The terms of ∂_c p move from the data into the shifted values of the
    // component, the derivative of each multi-index of f raised along c
    for (int component = 0; component < 2; ++component) {
        for (int b = 0; b + 2 <= m; ++b) {
            for (int a = 0; a + b + 2 <= m; ++a) {
                const int row = simplex_index(a + (component == 0 ? 1 : 0),
                                              b + (component == 1 ? 1 : 0),
                                              m - 1);
                const Eigen::VectorX<T> weights =
                    shift.source_weights.col(simplex_index(a, b, m - 2));

                for (int y = 0; y < m; ++y) {
                    for (int x = 0; x + y < m; ++x) {
                        shift.pressure_weights[component].col(
                            simplex_index(x, y, m)) +=
                            cartesian(row, simplex_index(x, y, m - 1))
                            * weights;
                    }
                }
            }
        }
    }

    return shift;
}

template StokesShift<double> stokes_shift(
    const Patch<TensorBSpline<double, 2>, 2>&,
    const BoundaryProjection<double>&, int, double);

} // namespace iguana
