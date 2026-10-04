/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "patch.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>

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

/// @brief Tensor-product B-spline of a B-spline basis, the basis itself
template<std::floating_point T, std::size_t d>
const TensorBSpline<T, d>& bspline_of(const TensorBSpline<T, d>& basis)
{
    return basis;
}

/// @brief Tensor-product B-spline that a NURBS basis weights
template<std::floating_point T, std::size_t d>
const TensorBSpline<T, d>& bspline_of(const TensorNURBS<T, d>& basis)
{
    return basis.bspline();
}

/// @brief Whether the weights of a basis are equal, as those of a B-spline
template<std::floating_point T, std::size_t d>
bool equal_weights(const TensorBSpline<T, d>&, T)
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

/// @brief Greville abscissae of a knot vector, the averages of the p knots
///        inside the support of each function, at which the control points
///        of the identity map lie
template<std::floating_point T>
Eigen::VectorX<T> greville_abscissae(const KnotVector<T>& knots)
{
    const std::vector<T>& values = knots.values();
    const int degree = knots.degree();
    const int count = static_cast<int>(values.size()) - degree - 1;

    Eigen::VectorX<T> result(count);

    for (int function = 0; function < count; ++function) {
        T sum = 0;

        for (int knot = 1; knot <= degree; ++knot)
            sum += values[static_cast<std::size_t>(function + knot)];

        result(function) = sum / static_cast<T>(degree);
    }

    return result;
}

/**
 * @brief Whether the map of a basis and its control points is affine,
 *        x = a + A xi, and the map if it is
 *
 * The map is read off the first control point and the last one along each
 * direction, then checked at every control point against its Greville
 * point, within rounding relative to the size of the patch. A direction of
 * degree zero, or a NURBS basis with unequal weights, does not reproduce
 * the identity, so its map is not affine
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

    if (!equal_weights(basis, rounding))
        return false;

    // Greville abscissae of each direction
    const TensorBSpline<Scalar, dim>& bspline = bspline_of(basis);
    std::array<Eigen::VectorX<Scalar>, dim> greville;
    std::array<int, dim> counts{};

    for (std::size_t direction = 0; direction < dim; ++direction) {
        const KnotVector<Scalar>& knots = bspline.axis(direction).knots();

        if (knots.degree() < 1)
            return false;

        greville[direction] = greville_abscissae(knots);
        counts[direction] = static_cast<int>(greville[direction].size());
    }

    const auto greville_point = [&greville, &counts](int function) {
        const std::array<int, dim> index = unflatten(function, counts);
        Eigen::Vector<Scalar, dim> point;

        for (std::size_t direction = 0; direction < dim; ++direction)
            point(direction) = greville[direction](index[direction]);

        return point;
    };

    // Candidate map, from the first control point and the last one along
    // each direction
    const Eigen::Vector<Scalar, n> first = coefficients.row(0).transpose();
    Eigen::Matrix<Scalar, n, dim> candidate;

    for (std::size_t direction = 0; direction < dim; ++direction) {
        std::array<int, dim> index{};
        index[direction] = counts[direction] - 1;

        const Eigen::Vector<Scalar, n> last =
            coefficients.row(flatten(index, counts)).transpose();
        const Scalar span = greville[direction](counts[direction] - 1)
                            - greville[direction](0);

        candidate.col(direction) = (last - first) / span;
    }

    const Eigen::Vector<Scalar, n> shift =
        first - candidate * greville_point(0);
    const Scalar size = (coefficients.colwise().maxCoeff()
                         - coefficients.colwise().minCoeff())
                            .maxCoeff();

    for (int function = 0; function < coefficients.rows(); ++function) {
        const Eigen::Vector<Scalar, n> image =
            shift + candidate * greville_point(function);

        if ((coefficients.row(function).transpose() - image)
                .cwiseAbs()
                .maxCoeff()
            > rounding * size)
            return false;
    }

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

// Curves in the plane and in space, planar regions, surfaces in space and
// volumes, on B-splines and on NURBS
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

} // namespace iguana
