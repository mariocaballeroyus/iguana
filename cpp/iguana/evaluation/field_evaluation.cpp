/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "field_evaluation.hpp"

#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include "iguana/assembly/element_values.hpp"
#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/grid/knot_vector.hpp"
#include "iguana/grid/tensor_grid.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Element of a grid holding a point, checked to lie in the grid up to
 *        round-off, so that a point on its edge passes
 *
 * @param grid Grid of the patch
 * @param parameters Parameters of the point, one per direction
 * @return Element index, in the numbering of the grid
 *
 * @throws std::invalid_argument If the point lies outside the grid
 */
template<typename T, std::size_t d, typename Parameters>
int element_holding_point(const TensorGrid<T, d>& grid,
                          const Parameters& parameters)
{
    std::array<int, d> index{};
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const KnotVector<T>& knots = grid.knots(direction);
        const T parameter = parameters(static_cast<Eigen::Index>(direction));
        const T rounding = 4096 * std::numeric_limits<T>::epsilon()
                           * (knots.domain_end() - knots.domain_start());

        if (!(parameter >= knots.domain_start() - rounding
              && parameter <= knots.domain_end() + rounding))
            throw std::invalid_argument("field_evaluation: "
                                        "the points must lie in the patch");

        index[direction] = element_holding(knots, parameter);
        counts[direction] = knots.num_elements();
    }

    return flatten(index, counts);
}

/**
 * @brief Field at points, from the matrix giving its value at each point out
 *        of the coefficients of the element holding it
 *
 * @param space Space of the field
 * @param patch Patch that maps the points
 * @param flags Optional values the matrix reads
 * @param coefficients Coefficients of the field, one per degree of freedom
 * @param points Points in physical space, one per row
 * @param local_matrix Fills the matrix of a point, given the values at it,
 *        its index and the matrix to overwrite
 * @return Value of the field at each point
 */
template<typename Basis, typename LocalMatrix>
Eigen::VectorX<typename Basis::Scalar> field_at_points(
    const MixedFunctionSpace<Basis>& space,
    const Patch<Basis, Basis::dimension>& patch, ValueFlags flags,
    const Eigen::VectorX<typename Basis::Scalar>& coefficients,
    const PointMatrix<typename Basis::Scalar, Basis::dimension>& points,
    const LocalMatrix& local_matrix)
{
    using Scalar = typename Basis::Scalar;
    constexpr std::size_t n = Basis::dimension;

    if (coefficients.size() != space.num_dofs())
        throw std::invalid_argument("field_evaluation: "
                                    "the coefficients must be one per degree "
                                    "of freedom");

    PointMatrix<Scalar, n> parameters;
    patch.invert_points(points, parameters);

    // Also checks that every field has the elements of the patch, whose grid
    // then locates the points in all of them
    ElementValues<Basis, n> values(patch, space.bases(), flags);

    Eigen::VectorX<Scalar> field(points.rows());
    Eigen::MatrixX<Scalar> matrix;
    std::vector<int> dofs;

    for (Eigen::Index point = 0; point < points.rows(); ++point) {
        const int element = element_holding_point(patch.basis().grid(),
                                                  parameters.row(point));

        space.dofs_on_element(element, dofs);

        if (dofs.empty())
            throw std::invalid_argument("field_evaluation: "
                                        "the points must not lie on cells "
                                        "outside the physical domain");

        values.reinit(element, parameters.row(point));
        local_matrix(values, point, matrix);

        if (matrix.rows() != std::ssize(dofs))
            throw std::invalid_argument("field_evaluation: "
                                        "the space must hold the unknowns of "
                                        "the element evaluated");

        field(point) = matrix.col(0).dot(coefficients(dofs));
    }

    return field;
}

} // namespace

template<typename Basis>
Eigen::VectorX<typename Basis::Scalar> trace_at_points(
    const MixedFunctionSpace<Basis>& space,
    const Patch<Basis, Basis::dimension>& patch,
    const Trace<Basis, Basis::dimension>& trace,
    const Eigen::VectorX<typename Basis::Scalar>& coefficients,
    const PointMatrix<typename Basis::Scalar, Basis::dimension>& points)
{
    using Scalar = typename Basis::Scalar;
    constexpr std::size_t n = Basis::dimension;

    return field_at_points(
        space, patch, trace.flags(), coefficients, points,
        [&trace](const ElementValues<Basis, n>& values, Eigen::Index,
                 Eigen::MatrixX<Scalar>& matrix) {
            trace.local_trace(values, matrix);
        });
}

template<typename Basis>
Eigen::VectorX<typename Basis::Scalar> flux_at_points(
    const MixedFunctionSpace<Basis>& space,
    const Patch<Basis, Basis::dimension>& patch,
    const Flux<Basis, Basis::dimension>& flux,
    const Eigen::VectorX<typename Basis::Scalar>& coefficients,
    const PointMatrix<typename Basis::Scalar, Basis::dimension>& points,
    const PointMatrix<typename Basis::Scalar, Basis::dimension>& normals)
{
    using Scalar = typename Basis::Scalar;
    constexpr std::size_t n = Basis::dimension;

    if (normals.rows() != points.rows())
        throw std::invalid_argument("field_evaluation: "
                                    "the normals must be one per point");

    PointMatrix<Scalar, n> normal(1, n);

    return field_at_points(
        space, patch, flux.flags(), coefficients, points,
        [&flux, &normals, &normal](const ElementValues<Basis, n>& values,
                                   Eigen::Index point,
                                   Eigen::MatrixX<Scalar>& matrix) {
            normal = normals.row(point);
            flux.local_flux(values, normal, matrix);
        });
}

template Eigen::VectorXd trace_at_points(
    const MixedFunctionSpace<TensorBSpline<double, 2>>&,
    const Patch<TensorBSpline<double, 2>, 2>&,
    const Trace<TensorBSpline<double, 2>, 2>&, const Eigen::VectorXd&,
    const PointMatrix<double, 2>&);
template Eigen::VectorXd trace_at_points(
    const MixedFunctionSpace<TensorBSpline<double, 3>>&,
    const Patch<TensorBSpline<double, 3>, 3>&,
    const Trace<TensorBSpline<double, 3>, 3>&, const Eigen::VectorXd&,
    const PointMatrix<double, 3>&);
template Eigen::VectorXd trace_at_points(
    const MixedFunctionSpace<TensorNURBS<double, 2>>&,
    const Patch<TensorNURBS<double, 2>, 2>&,
    const Trace<TensorNURBS<double, 2>, 2>&, const Eigen::VectorXd&,
    const PointMatrix<double, 2>&);
template Eigen::VectorXd trace_at_points(
    const MixedFunctionSpace<TensorNURBS<double, 3>>&,
    const Patch<TensorNURBS<double, 3>, 3>&,
    const Trace<TensorNURBS<double, 3>, 3>&, const Eigen::VectorXd&,
    const PointMatrix<double, 3>&);

template Eigen::VectorXd flux_at_points(
    const MixedFunctionSpace<TensorBSpline<double, 2>>&,
    const Patch<TensorBSpline<double, 2>, 2>&,
    const Flux<TensorBSpline<double, 2>, 2>&, const Eigen::VectorXd&,
    const PointMatrix<double, 2>&, const PointMatrix<double, 2>&);
template Eigen::VectorXd flux_at_points(
    const MixedFunctionSpace<TensorBSpline<double, 3>>&,
    const Patch<TensorBSpline<double, 3>, 3>&,
    const Flux<TensorBSpline<double, 3>, 3>&, const Eigen::VectorXd&,
    const PointMatrix<double, 3>&, const PointMatrix<double, 3>&);
template Eigen::VectorXd flux_at_points(
    const MixedFunctionSpace<TensorNURBS<double, 2>>&,
    const Patch<TensorNURBS<double, 2>, 2>&,
    const Flux<TensorNURBS<double, 2>, 2>&, const Eigen::VectorXd&,
    const PointMatrix<double, 2>&, const PointMatrix<double, 2>&);
template Eigen::VectorXd flux_at_points(
    const MixedFunctionSpace<TensorNURBS<double, 3>>&,
    const Patch<TensorNURBS<double, 3>, 3>&,
    const Flux<TensorNURBS<double, 3>, 3>&, const Eigen::VectorXd&,
    const PointMatrix<double, 3>&, const PointMatrix<double, 3>&);

} // namespace iguana
