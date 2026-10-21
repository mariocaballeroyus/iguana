/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;
using iguana::Patch;
using iguana::PointMatrix;
using iguana::TensorBSpline;
using iguana::simplex_index;
using iguana::simplex_size;

/// @brief Basis quadratic along the first direction and cubic along the
///        second, with interior knots in both
TensorBSpline<double, 2> basis()
{
    return TensorBSpline<double, 2>(
        {BSpline<double>(2, {0., 0., 0., .3, .7, 1., 1., 1.}),
         BSpline<double>(3, {0., 0., 0., 0., .4, 1., 1., 1., 1.})});
}

/// @brief Sheared and rotated affine map, x = a + A xi
Eigen::Matrix2d linear()
{
    Eigen::Matrix2d map;
    map << 2., .5,
           -.75, 1.5;

    return map;
}

const Eigen::Vector2d offset(.5, -1.);

/// @brief Patch of the affine map over basis(), with its control points at
///        the images of the Greville points
Patch<TensorBSpline<double, 2>, 2> sheared()
{
    const TensorBSpline<double, 2> splines = basis();
    PointMatrix<double, 2> coefficients(splines.num_functions(), 2);
    const Eigen::VectorXd first =
        iguana::greville_abscissae(splines.axis(0).knots());
    const Eigen::VectorXd second =
        iguana::greville_abscissae(splines.axis(1).knots());

    for (Eigen::Index j = 0; j < second.size(); ++j)
        for (Eigen::Index i = 0; i < first.size(); ++i)
            coefficients.row(j * first.size() + i) =
                offset + linear() * Eigen::Vector2d(first(i), second(j));

    return {splines, coefficients};
}

/// @brief Derivative d_x^a d_y^b at a point of the polynomial with
///        coefficient c_ij of x^i y^j
double derivative(const Eigen::MatrixXd& coefficients, int a, int b,
                  const Eigen::RowVector2d& point)
{
    double sum = 0.;

    for (Eigen::Index i = a; i < coefficients.rows(); ++i) {
        for (Eigen::Index j = b; j < coefficients.cols(); ++j) {
            double factor = coefficients(i, j);

            for (Eigen::Index k = 0; k < a; ++k)
                factor *= static_cast<double>(i - k);

            for (Eigen::Index k = 0; k < b; ++k)
                factor *= static_cast<double>(j - k);

            sum += factor * std::pow(point(0), static_cast<double>(i - a))
                   * std::pow(point(1), static_cast<double>(j - b));
        }
    }

    return sum;
}

} // namespace

TEST_CASE("A field interpolated in the basis of a patch reproduces "
          "polynomials of its degree with their Cartesian derivatives",
          "[field_interpolation]")
{
    const Patch<TensorBSpline<double, 2>, 2> patch = sheared();
    const int degree = 2;

    // A random polynomial of total degree two in x and y, which the
    // quadratic by cubic splines span through the affine map
    std::mt19937 generator(7);
    std::uniform_real_distribution<double> uniform(-1., 1.);
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(degree + 1,
                                                         degree + 1);

    for (int i = 0; i <= degree; ++i)
        for (int j = 0; i + j <= degree; ++j)
            coefficients(i, j) = uniform(generator);

    const PointMatrix<double, 2> nodes = iguana::interpolation_points(patch);
    Eigen::VectorXd values(nodes.rows());

    for (Eigen::Index node = 0; node < nodes.rows(); ++node)
        values(node) = derivative(coefficients, 0, 0, nodes.row(node));

    // Points at random parameters of the patch, every element included
    std::uniform_real_distribution<double> unit(0., 1.);
    PointMatrix<double, 2> points(20, 2);

    for (Eigen::Index point = 0; point < points.rows(); ++point)
        points.row(point) =
            offset + linear() * Eigen::Vector2d(unit(generator),
                                                unit(generator));

    const Eigen::MatrixXd derivs =
        iguana::interpolated_derivs(patch, values, points, degree);

    REQUIRE(derivs.rows() == points.rows());
    REQUIRE(derivs.cols() == simplex_size(degree));

    for (Eigen::Index point = 0; point < points.rows(); ++point) {
        for (int b = 0; b <= degree; ++b) {
            for (int a = 0; a + b <= degree; ++a) {
                INFO("point " << point << ", a = " << a << ", b = " << b);
                REQUIRE(std::abs(derivs(point, simplex_index(a, b, degree))
                                 - derivative(coefficients, a, b,
                                              points.row(point)))
                        < 1e-11);
            }
        }
    }
}

TEST_CASE("Interpolated fields need an affine map, one value per function "
          "and an order within the degree", "[field_interpolation]")
{
    const Patch<TensorBSpline<double, 2>, 2> patch = sheared();
    const Eigen::VectorXd values =
        Eigen::VectorXd::Zero(patch.basis().num_functions());
    const PointMatrix<double, 2> points = patch.coefficients().topRows(2);

    // The quadratic direction has no third derivatives
    REQUIRE_THROWS_AS(iguana::interpolated_derivs(patch, values, points, 3),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::interpolated_derivs(patch, values, points, -1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::interpolated_derivs(
                          patch, Eigen::VectorXd(values.head(3)), points, 1),
                      std::invalid_argument);

    // A control point moved off the affine image of its Greville point
    PointMatrix<double, 2> moved = patch.coefficients();
    moved(4, 0) += .1;
    const Patch<TensorBSpline<double, 2>, 2> curved(patch.basis(), moved);

    REQUIRE_THROWS_AS(iguana::interpolation_points(curved),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::interpolated_derivs(curved, values, points, 1),
                      std::invalid_argument);
}
