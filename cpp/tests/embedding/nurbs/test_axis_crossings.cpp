/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/embedding/nurbs/axis_crossings.hpp>
#include <iguana/grid/knot_vector.hpp>

#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::PointMatrix;
using iguana::TensorBSpline;
using iguana::TensorNURBS;

using Curve = iguana::Patch<TensorNURBS<double, 1>, 2>;

constexpr double tolerance = 1e-12;

/// @brief Quadratic NURBS curve of one element
Curve arc(const PointMatrix<double, 2>& points,
          const Eigen::Vector3d& weights)
{
    const TensorBSpline<double, 1> bspline(
        {BSpline<double>(2, {0., 0., 0., 1., 1., 1.})});

    return {TensorNURBS<double, 1>(bspline, weights), points};
}

/// @brief Quarter of the unit circle from (1, 0) to (0, 1)
Curve quarter_circle()
{
    PointMatrix<double, 2> points(3, 2);
    points << 1., 0.,
              1., 1.,
              0., 1.;

    return arc(points, Eigen::Vector3d(1., std::sqrt(.5), 1.));
}

/// @brief Homogeneous Bezier points of the element of a curve
Eigen::MatrixXd bezier_points(const Curve& curve)
{
    const Eigen::VectorXd& weights = curve.basis().weights();

    Eigen::MatrixXd homogeneous(weights.size(), 3);
    homogeneous << weights.asDiagonal() * curve.coefficients(), weights;

    return iguana::extraction_operators(
               curve.basis().bspline().axis(0).knots())[0]
               .transpose()
           * homogeneous;
}

/// @brief Point of an element at a parameter, from its homogeneous Bezier
///        points
Eigen::Vector2d point_at(const Eigen::MatrixXd& points, double t)
{
    const double weight = iguana::de_casteljau<double>(points.col(2), t);

    return {iguana::de_casteljau<double>(points.col(0), t) / weight,
            iguana::de_casteljau<double>(points.col(1), t) / weight};
}

} // namespace

TEST_CASE("A quarter circle crosses each line through it once, on the "
          "circle", "[axis_crossings]")
{
    const Eigen::MatrixXd points = bezier_points(quarter_circle());

    for (std::size_t axis = 0; axis < 2; ++axis) {
        const std::vector<double> crossings =
            iguana::curve_crossings(points, axis, .5, tolerance);

        REQUIRE(crossings.size() == 1);

        const Eigen::Vector2d point = point_at(points, crossings[0]);

        REQUIRE_THAT(point(axis), WithinAbs(.5, 1e-15));
        REQUIRE_THAT(point.norm(), WithinAbs(1., 1e-15));
    }
}

TEST_CASE("Lines a curve touches, misses or meets at an end are not "
          "crossed", "[axis_crossings]")
{
    // A parabola rising from y = 1/2 to touch y = 1 at t = 1/2
    PointMatrix<double, 2> touching(3, 2);
    touching << 0., .5,
                .5, 1.5,
                1., .5;

    const Eigen::MatrixXd parabola =
        bezier_points(arc(touching, Eigen::Vector3d::Ones()));
    const Eigen::MatrixXd circle = bezier_points(quarter_circle());

    REQUIRE(iguana::curve_crossings(parabola, 1, 1., tolerance).empty());
    REQUIRE(iguana::curve_crossings(circle, 0, 1.5, tolerance).empty());
    REQUIRE(iguana::curve_crossings(circle, 1, 0., tolerance).empty());
}

TEST_CASE("An element lying on a line shares its coordinate and does not "
          "cross it", "[axis_crossings]")
{
    PointMatrix<double, 2> points(3, 2);
    points << .3, .1,
              .3, .2,
              .3, .4;

    const Curve line = arc(points, Eigen::Vector3d(1., 2., 1.));

    REQUIRE(iguana::shared_coordinate(line, 0, 0) == std::optional(.3));
    REQUIRE_FALSE(iguana::shared_coordinate(line, 0, 1));
    REQUIRE_FALSE(iguana::shared_coordinate(quarter_circle(), 0, 0));

    REQUIRE(iguana::curve_crossings(bezier_points(line), 0, .3, tolerance)
                .empty());
}
