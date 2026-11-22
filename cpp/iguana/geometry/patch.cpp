/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "patch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>
#include <Eigen/QR>

#include "iguana/grid/knot_vector.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

using Eigen::placeholders::all;

namespace
{

/// @brief Jacobian of a patch at a point, with one column per parametric
///        direction
template<typename Scalar, std::size_t n, std::size_t d>
Eigen::Matrix<Scalar, n, d> jacobian_at(
    const std::array<PointMatrix<Scalar, n>, d>& tangents,
    Eigen::Index point)
{
    Eigen::Matrix<Scalar, n, d> jacobian;

    for (std::size_t direction = 0; direction < d; ++direction)
        jacobian.col(direction) = tangents[direction].row(point).transpose();

    return jacobian;
}

/**
 * @brief Greville points of a tensor-product B-spline basis, one per
 *        function in its numbering, the products of the abscissae of its
 *        directions
 */
template<std::floating_point T, std::size_t d>
Eigen::MatrixX<T> greville_points(const TensorBSpline<T, d>& basis)
{
    std::array<Eigen::VectorX<T>, d> abscissae;
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        abscissae[direction] =
            greville_abscissae(basis.axis(direction).knots());
        counts[direction] = static_cast<int>(abscissae[direction].size());
    }

    Eigen::MatrixX<T> points(basis.num_functions(), d);

    for (int function = 0; function < basis.num_functions(); ++function) {
        const std::array<int, d> index = unflatten(function, counts);

        for (std::size_t direction = 0; direction < d; ++direction)
            points(function, direction) =
                abscissae[direction](index[direction]);
    }

    return points;
}

/// @brief Greville points of a NURBS basis, those of its B-splines
template<std::floating_point T, std::size_t d>
Eigen::MatrixX<T> greville_points(const TensorNURBS<T, d>& basis)
{
    return greville_points(basis.bspline());
}

/// @brief Greville points of a hierarchical basis, those of each function
///        in the tensor basis of its level
template<std::floating_point T, std::size_t d>
Eigen::MatrixX<T> greville_points(const THBSpline<T, d>& basis)
{
    Eigen::MatrixX<T> points(basis.num_functions(), d);

    for (int level = 0; level < basis.num_levels(); ++level) {
        const Eigen::MatrixX<T> level_points =
            greville_points(basis.level(level));
        const std::vector<int>& active = basis.active_functions(level);

        for (std::size_t k = 0; k < active.size(); ++k)
            points.row(basis.offset(level) + static_cast<Eigen::Index>(k)) =
                level_points.row(active[k]);
    }

    return points;
}

/// @brief Lowest degree of a basis over its directions
template<std::floating_point T, std::size_t d>
int min_degree(const TensorBSpline<T, d>& basis)
{
    int degree = basis.axis(0).degree();

    for (std::size_t direction = 1; direction < d; ++direction)
        degree = std::min(degree, basis.axis(direction).degree());

    return degree;
}

template<std::floating_point T, std::size_t d>
int min_degree(const TensorNURBS<T, d>& basis)
{
    return min_degree(basis.bspline());
}

template<std::floating_point T, std::size_t d>
int min_degree(const THBSpline<T, d>& basis)
{
    return std::ranges::min(basis.degrees());
}

/// @brief Whether the weights of a basis are equal, as those of a B-spline
template<std::floating_point T, std::size_t d>
bool equal_weights(const TensorBSpline<T, d>&, T)
{
    return true;
}

template<std::floating_point T, std::size_t d>
bool equal_weights(const THBSpline<T, d>&, T)
{
    return true;
}

/// @brief Whether the weights of a NURBS basis are equal up to a relative
///        tolerance, so that its functions are B-splines
template<std::floating_point T, std::size_t d>
bool equal_weights(const TensorNURBS<T, d>& basis, T tolerance)
{
    const Eigen::VectorX<T>& weights = basis.weights();

    return (weights.array() - weights(0)).abs().maxCoeff()
           <= tolerance * std::abs(weights(0));
}

/**
 * @brief Whether the map of a basis and its control points is affine,
 *        x = a + A xi, and the map if it is
 *
 * An affine map has the images of the Greville points as control points,
 * so the map is fitted to the control points at the Greville points by
 * least squares, then checked at every one of them within rounding
 * relative to the size of the patch. A direction of degree zero, or a
 * NURBS basis with unequal weights, does not reproduce the identity, so
 * its map is not affine
 *
 * @param offset Output offset a, left zero if the map is not affine
 * @param linear Output linear part A, left zero if the map is not affine
 */
template<typename Basis, std::size_t n>
bool affine_map(
    const Basis& basis,
    const PointMatrix<typename Basis::Scalar, n>& coefficients,
    Eigen::Vector<typename Basis::Scalar, n>& offset,
    Eigen::Matrix<typename Basis::Scalar, n, Basis::dimension>& linear)
{
    using Scalar = typename Basis::Scalar;
    constexpr std::size_t dim = Basis::dimension;

    offset.setZero();
    linear.setZero();

    const Scalar rounding = 4096 * std::numeric_limits<Scalar>::epsilon();

    if (!equal_weights(basis, rounding) || min_degree(basis) < 1)
        return false;

    // The fit c_i ≈ a + A g_i, a column of unknowns per coordinate
    Eigen::MatrixX<Scalar> system(coefficients.rows(), dim + 1);
    system.col(0).setOnes();
    system.rightCols(dim) = greville_points(basis);

    const Eigen::MatrixX<Scalar> fit =
        system.colPivHouseholderQr().solve(coefficients);
    const Eigen::Vector<Scalar, n> shift = fit.row(0).transpose();
    const Eigen::Matrix<Scalar, n, dim> candidate =
        fit.bottomRows(dim).transpose();

    // The size of the patch, or its distance from the origin for a map
    // collapsed to a point, whose fit still rounds
    const Scalar size = std::max((coefficients.colwise().maxCoeff()
                                  - coefficients.colwise().minCoeff())
                                     .maxCoeff(),
                                 coefficients.cwiseAbs().maxCoeff());
    const Eigen::MatrixX<Scalar> images =
        (system.rightCols(dim) * candidate.transpose()).rowwise()
        + shift.transpose();

    if ((coefficients - images).cwiseAbs().maxCoeff() > rounding * size)
        return false;

    offset = shift;
    linear = candidate;

    return true;
}

} // namespace

template<typename Basis, std::size_t n>
Patch<Basis, n>::Patch(Basis basis, PointMatrix<Scalar, n> coefficients)
    : basis_(std::move(basis)),
      coefficients_(std::move(coefficients))
{
    if (coefficients_.rows() != basis_.num_functions())
        throw std::invalid_argument("Patch: "
                                    "there must be one control point per "
                                    "basis function");

    affine_ = affine_map<Basis, n>(basis_, coefficients_, offset_, linear_);
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::position_on_element(
    const Eigen::VectorXi& actives, const Eigen::MatrixX<Scalar>& values,
    PointMatrix<Scalar, n>& positions) const
{
    // Reuse the output buffer when its shape is unchanged
    positions.resize(values.cols(), n);

    // Gather the control points of the active functions, then weight each
    // of them by the value of its function at every point
    positions.noalias() = values.transpose() * coefficients_(actives, all);
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::tangent_on_element(
    const Eigen::VectorXi& actives,
    const std::array<Eigen::MatrixX<Scalar>, dim>& gradients,
    std::array<PointMatrix<Scalar, n>, dim>& tangents) const
{
    for (std::size_t direction = 0; direction < dim; ++direction) {
        // Weight the control points of the active functions by the
        // derivatives of their functions along the direction
        tangents[direction].noalias() =
            gradients[direction].transpose() * coefficients_(actives, all);
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::second_tangent_on_element(
    const Eigen::VectorXi& actives,
    const std::array<Eigen::MatrixX<Scalar>, dim * (dim + 1) / 2>& hessians,
    std::array<PointMatrix<Scalar, n>, dim * (dim + 1) / 2>& second_tangents)
    const
{
    for (std::size_t pair = 0; pair < hessians.size(); ++pair) {
        // Weight the control points of the active functions by the second
        // derivatives of their functions along the pair
        second_tangents[pair].noalias() =
            hessians[pair].transpose() * coefficients_(actives, all);
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::measure_on_element(
    const std::array<PointMatrix<Scalar, n>, dim>& tangents,
    Eigen::VectorX<Scalar>& measures)
{
    measures.resize(tangents[0].rows());

    for (Eigen::Index point = 0; point < measures.size(); ++point) {
        const Eigen::Matrix<Scalar, n, dim> jacobian =
            jacobian_at<Scalar, n, dim>(tangents, point);

        if constexpr (dim == n)
            measures(point) = std::abs(jacobian.determinant());
        else if constexpr (dim < n)
            measures(point) =
                std::sqrt((jacobian.transpose() * jacobian).determinant());
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::boundary_measure_on_element(
    const std::array<PointMatrix<Scalar, n>, dim>& tangents,
    const Eigen::MatrixX<Scalar>& normals,
    Eigen::VectorX<Scalar>& measures)
{
    measures.resize(normals.rows());

    for (Eigen::Index point = 0; point < measures.size(); ++point) {
        const Eigen::Matrix<Scalar, n, dim> jacobian =
            jacobian_at<Scalar, n, dim>(tangents, point);
        const Eigen::Matrix<Scalar, dim, dim> metric =
            jacobian.transpose() * jacobian;

        // ds = sqrt(det G) |J G^-1 m| dŝ, Nanson's formula with the metric
        const Eigen::Vector<Scalar, n> scaled =
            jacobian * metric.inverse() * normals.row(point).transpose();

        measures(point) = std::sqrt(metric.determinant()) * scaled.norm();
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::physical_normal_on_element(
    const std::array<PointMatrix<Scalar, n>, dim>& tangents,
    const Eigen::MatrixX<Scalar>& normals,
    PointMatrix<Scalar, n>& physical_normals)
{
    physical_normals.resize(normals.rows(), n);

    for (Eigen::Index point = 0; point < normals.rows(); ++point) {
        const Eigen::Matrix<Scalar, n, dim> jacobian =
            jacobian_at<Scalar, n, dim>(tangents, point);
        const Eigen::Matrix<Scalar, dim, dim> metric =
            jacobian.transpose() * jacobian;

        // n ds = sqrt(det G) J G^-1 m dŝ, so the normal points along J G^-1 m
        const Eigen::Vector<Scalar, n> scaled =
            jacobian * metric.inverse() * normals.row(point).transpose();

        physical_normals.row(point) = scaled.normalized().transpose();
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::physical_grad_on_element(
    const std::array<PointMatrix<Scalar, n>, dim>& tangents,
    const std::array<Eigen::MatrixX<Scalar>, dim>& gradients,
    std::array<Eigen::MatrixX<Scalar>, n>& physical_gradients)
    requires (dim == n)
{
    const Eigen::Index num_active = gradients[0].rows();
    const Eigen::Index num_points = gradients[0].cols();

    for (Eigen::MatrixX<Scalar>& physical : physical_gradients)
        physical.resize(num_active, num_points);

    for (Eigen::Index point = 0; point < num_points; ++point) {
        // Component j of J^-T grad_xi is the sum of (J^-1)_kj d/dxi_k
        const Eigen::Matrix<Scalar, n, n> inverse =
            jacobian_at<Scalar, n, dim>(tangents, point).inverse();

        for (std::size_t component = 0; component < n; ++component) {
            physical_gradients[component].col(point).setZero();

            for (std::size_t direction = 0; direction < dim; ++direction)
                physical_gradients[component].col(point) +=
                    inverse(direction, component) *
                    gradients[direction].col(point);
        }
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::physical_hess_on_element(
    const std::array<PointMatrix<Scalar, n>, dim>& tangents,
    const std::array<PointMatrix<Scalar, n>, dim * (dim + 1) / 2>&
        second_tangents,
    const std::array<Eigen::MatrixX<Scalar>, n>& physical_gradients,
    const std::array<Eigen::MatrixX<Scalar>, dim * (dim + 1) / 2>& hessians,
    std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>& physical_hessians)
    requires (dim == n)
{
    const Eigen::Index num_active = hessians[0].rows();
    const Eigen::Index num_points = hessians[0].cols();

    for (Eigen::MatrixX<Scalar>& physical : physical_hessians)
        physical.resize(num_active, num_points);

    // Second derivatives of the functions along each pair of directions,
    // less the curvature of the map, at one point
    std::array<Eigen::VectorX<Scalar>, dim * (dim + 1) / 2> corrected;

    for (Eigen::Index point = 0; point < num_points; ++point) {
        const Eigen::Matrix<Scalar, n, n> inverse =
            jacobian_at<Scalar, n, dim>(tangents, point).inverse();

        // ∂²N/∂ξ_a∂ξ_b − Σ_k ∂N/∂x_k ∂²x_k/∂ξ_a∂ξ_b
        for (std::size_t pair = 0; pair < corrected.size(); ++pair) {
            corrected[pair] = hessians[pair].col(point);

            for (std::size_t coordinate = 0; coordinate < n; ++coordinate)
                corrected[pair] -= second_tangents[pair](point, coordinate)
                                   * physical_gradients[coordinate].col(point);
        }

        // Σ_ab (J⁻¹)_ai (J⁻¹)_bj over both orders of each mixed pair
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i; j < n; ++j) {
                Eigen::MatrixX<Scalar>& physical =
                    physical_hessians[pair_index<n>(i, j)];
                physical.col(point).setZero();

                for (std::size_t a = 0; a < dim; ++a) {
                    for (std::size_t b = 0; b < dim; ++b)
                        physical.col(point) +=
                            inverse(a, i) * inverse(b, j)
                            * corrected[pair_index<dim>(a, b)];
                }
            }
        }
    }
}

template<typename Basis, std::size_t n>
void Patch<Basis, n>::invert_points(const PointMatrix<Scalar, n>& points,
                                    PointMatrix<Scalar, dim>& parameters) const
    requires (dim == n)
{
    if (!affine_)
        throw std::invalid_argument("Patch: "
                                    "the map must be affine to be inverted");

    // Reuse the output buffer when its shape is unchanged
    parameters.resize(points.rows(), dim);

    // The product is evaluated before the assignment, so the points may be
    // inverted in place
    parameters = (points.rowwise() - offset_.transpose())
                 * linear_.inverse().transpose();
}

template<std::floating_point T, std::size_t d, std::size_t n>
Patch<THBSpline<T, d>, n> refine(const Patch<TensorBSpline<T, d>, n>& patch,
                                 HierarchicalGrid<T, d> grid)
{
    std::array<int, d> degrees{};

    for (std::size_t direction = 0; direction < d; ++direction)
        degrees[direction] = patch.basis().axis(direction).degree();

    THBSpline<T, d> basis(std::move(grid), degrees);

    if (basis.level(0) != patch.basis())
        throw std::invalid_argument("Patch: "
                                    "the coarse level of the grid must have "
                                    "the knots of the patch");

    // The truncated functions preserve the coefficients of the map
    const PointMatrix<T, n> coefficients =
        basis.coefficients(patch.coefficients());

    return Patch<THBSpline<T, d>, n>(std::move(basis), coefficients);
}

// Curves in the plane and in space, planar regions, surfaces in space and
// volumes, on B-splines and on NURBS, and the regions, surfaces and
// volumes refined on hierarchical B-splines
template class Patch<TensorBSpline<double, 1>, 2>;
template class Patch<TensorBSpline<double, 1>, 3>;
template class Patch<TensorBSpline<double, 2>, 2>;
template class Patch<TensorBSpline<double, 2>, 3>;
template class Patch<TensorBSpline<double, 3>, 3>;
template class Patch<TensorNURBS<double, 1>, 2>;
template class Patch<TensorNURBS<double, 1>, 3>;
template class Patch<TensorNURBS<double, 2>, 2>;
template class Patch<TensorNURBS<double, 2>, 3>;
template class Patch<TensorNURBS<double, 3>, 3>;
template class Patch<THBSpline<double, 2>, 2>;
template class Patch<THBSpline<double, 2>, 3>;
template class Patch<THBSpline<double, 3>, 3>;

template Patch<THBSpline<double, 2>, 2> refine(
    const Patch<TensorBSpline<double, 2>, 2>&, HierarchicalGrid<double, 2>);
template Patch<THBSpline<double, 2>, 3> refine(
    const Patch<TensorBSpline<double, 2>, 3>&, HierarchicalGrid<double, 2>);
template Patch<THBSpline<double, 3>, 3> refine(
    const Patch<TensorBSpline<double, 3>, 3>&, HierarchicalGrid<double, 3>);

} // namespace iguana
