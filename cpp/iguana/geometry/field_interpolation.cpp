/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "field_interpolation.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include <Eigen/LU>

#include "iguana/geometry/directional_derivs.hpp"
#include "iguana/grid/knot_vector.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<std::floating_point T>
Eigen::MatrixX<T> collocation_matrix(const BSpline<T>& bspline)
{
    const KnotVector<T>& knots = bspline.knots();
    const Eigen::VectorX<T> abscissae = greville_abscissae(knots);
    Eigen::MatrixX<T> matrix =
        Eigen::MatrixX<T>::Zero(abscissae.size(), bspline.num_functions());
    Eigen::MatrixX<T> values;

    for (Eigen::Index row = 0; row < abscissae.size(); ++row) {
        const T abscissa = abscissae(row);
        const int first =
            bspline.first_active(element_holding(knots, abscissa));

        bspline.eval_on_element(first, std::span<const T>(&abscissa, 1),
                                values);
        matrix.block(row, first, 1, values.rows()) = values.transpose();
    }

    return matrix;
}

template<std::floating_point T>
PointMatrix<T, 2> interpolation_points(
    const Patch<TensorBSpline<T, 2>, 2>& patch)
{
    if (!patch.is_affine())
        throw std::invalid_argument("interpolation_points: "
                                    "the map of the patch must be affine");

    const TensorBSpline<T, 2>& basis = patch.basis();
    const Eigen::VectorX<T> first = greville_abscissae(basis.axis(0).knots());
    const Eigen::VectorX<T> second =
        greville_abscissae(basis.axis(1).knots());

    // x = a + A ξ at the Greville points, the first direction running
    // fastest as the functions do
    PointMatrix<T, 2> points(first.size() * second.size(), 2);

    for (Eigen::Index j = 0; j < second.size(); ++j) {
        for (Eigen::Index i = 0; i < first.size(); ++i) {
            const Eigen::Vector2<T> greville(first(i), second(j));
            points.row(j * first.size() + i) =
                patch.offset() + patch.linear_part() * greville;
        }
    }

    return points;
}

template<std::floating_point T>
Eigen::MatrixX<T> interpolated_derivs(
    const Patch<TensorBSpline<T, 2>, 2>& patch,
    const Eigen::VectorX<T>& values, const PointMatrix<T, 2>& points,
    int order)
{
    const TensorBSpline<T, 2>& basis = patch.basis();
    const BSpline<T>& first = basis.axis(0);
    const BSpline<T>& second = basis.axis(1);

    if (order < 0)
        throw std::invalid_argument("interpolated_derivs: "
                                    "the order must be non-negative");

    if (order > std::min(first.degree(), second.degree()))
        throw std::invalid_argument("interpolated_derivs: "
                                    "the order must not exceed the degree of "
                                    "the basis in any direction");

    if (!patch.is_affine())
        throw std::invalid_argument("interpolated_derivs: "
                                    "the map of the patch must be affine");

    if (values.size() != basis.num_functions())
        throw std::invalid_argument("interpolated_derivs: "
                                    "the values must be one per function of "
                                    "the basis");

    // Coefficients C of the interpolant, B_ξ C B_η^T = V with the values V
    // at the Greville points, the first direction along the rows
    const Eigen::Map<const Eigen::MatrixX<T>> grid(
        values.data(), first.num_functions(), second.num_functions());
    const Eigen::MatrixX<T> partial =
        collocation_matrix(first).partialPivLu().solve(grid);
    const Eigen::MatrixX<T> coefficients =
        collocation_matrix(second)
            .partialPivLu()
            .solve(partial.transpose())
            .transpose();

    // ∂_x and ∂_y are the parametric derivatives along A^−1 e_x and A^−1 e_y,
    // the same at every point of an affine map
    const Eigen::Matrix2<T> inverse = patch.linear_part().inverse();
    const Eigen::MatrixX<T> cartesian = directional_derivs<T>(inverse, order);

    PointMatrix<T, 2> parameters;
    patch.invert_points(points, parameters);

    Eigen::MatrixX<T> derivs(points.rows(), simplex_size(order));
    Eigen::VectorX<T> parametric(simplex_size(order));
    std::vector<Eigen::MatrixX<T>> first_derivs;
    std::vector<Eigen::MatrixX<T>> second_derivs;

    for (Eigen::Index point = 0; point < points.rows(); ++point) {
        const T xi = parameters(point, 0);
        const T eta = parameters(point, 1);
        const int first_active =
            first.first_active(element_holding(first.knots(), xi));
        const int second_active =
            second.first_active(element_holding(second.knots(), eta));

        first.derivs_on_element(first_active, std::span<const T>(&xi, 1),
                                order, first_derivs);
        second.derivs_on_element(second_active, std::span<const T>(&eta, 1),
                                 order, second_derivs);

        // ∂_ξ^a ∂_η^b f_h = Σ_ij c_ij N_i^(a)(ξ) M_j^(b)(η) over the functions
        // active on the element
        const Eigen::MatrixX<T> active = coefficients.block(
            first_active, second_active, first.degree() + 1,
            second.degree() + 1);

        for (int b = 0; b <= order; ++b) {
            for (int a = 0; a + b <= order; ++a) {
                parametric(simplex_index(a, b, order)) =
                    (first_derivs[static_cast<std::size_t>(a)].transpose()
                     * active * second_derivs[static_cast<std::size_t>(b)])
                        .value();
            }
        }

        derivs.row(point) = (cartesian * parametric).transpose();
    }

    return derivs;
}

template Eigen::MatrixXd collocation_matrix(const BSpline<double>&);
template PointMatrix<double, 2> interpolation_points(
    const Patch<TensorBSpline<double, 2>, 2>&);
template Eigen::MatrixXd interpolated_derivs(
    const Patch<TensorBSpline<double, 2>, 2>&, const Eigen::VectorXd&,
    const PointMatrix<double, 2>&, int);

} // namespace iguana
