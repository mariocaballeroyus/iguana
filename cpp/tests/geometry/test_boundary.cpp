/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;
using iguana::Boundary;
using iguana::PointMatrix;
using iguana::TensorBSpline;
using iguana::TensorNURBS;

constexpr double radius = 2.;

constexpr double pi = std::numbers::pi;

/// @brief Circle of the radius around the origin, counterclockwise from
///        (radius, 0), as four quadratic arcs joined at double knots
Boundary<double, 2>::Face circle()
{
    const double corner = std::sqrt(.5);
    const TensorBSpline<double, 1> bspline({BSpline<double>(
        2, {0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.})});

    Eigen::VectorXd weights(9);
    weights << 1., corner, 1., corner, 1., corner, 1., corner, 1.;

    PointMatrix<double, 2> points(9, 2);
    points << 1., 0.,
              1., 1.,
              0., 1.,
             -1., 1.,
             -1., 0.,
             -1., -1.,
              0., -1.,
              1., -1.,
              1., 0.;

    return {TensorNURBS<double, 1>(bspline, weights), radius * points};
}

/// @brief Half of a cylinder of the radius around the z axis, of unit
///        height, with the angle along the first direction and the height
///        along the second
Boundary<double, 3>::Face half_cylinder()
{
    const double corner = std::sqrt(.5);
    const TensorBSpline<double, 2> bspline(
        {BSpline<double>(2, {0., 0., 0., .5, .5, 1., 1., 1.}),
         BSpline<double>(1, {0., 0., 1., 1.})});

    Eigen::VectorXd weights(10);
    weights << 1., corner, 1., corner, 1.,
               1., corner, 1., corner, 1.;

    PointMatrix<double, 3> points(10, 3);
    points << radius, 0., 0.,
              radius, radius, 0.,
              0., radius, 0.,
             -radius, radius, 0.,
             -radius, 0., 0.,
              radius, 0., 1.,
              radius, radius, 1.,
              0., radius, 1.,
             -radius, radius, 1.,
             -radius, 0., 1.;

    return {TensorNURBS<double, 2>(bspline, weights), points};
}

/// @brief Straight face from one point to another, a line of degree one
Boundary<double, 2>::Face segment(const Eigen::RowVector2d& from,
                                  const Eigen::RowVector2d& to)
{
    PointMatrix<double, 2> points(2, 2);
    points << from, to;

    const TensorBSpline<double, 1> line(
        {BSpline<double>(1, {0., 0., 1., 1.})});

    return {TensorNURBS<double, 1>(line, Eigen::Vector2d::Ones()), points};
}

/**
 * @brief Largest difference between the normals of a face and its radial
 *        directions times its sign, at points of every element of the face
 *
 * The radial direction leaves the origin in the plane and the z axis in
 * space, which is the outward normal of the circle and of the cylinder
 */
template<std::size_t n>
double radial_error(const Boundary<double, n>& boundary, int face)
{
    constexpr std::size_t d = n - 1;
    const typename Boundary<double, n>::Face& patch = boundary.face(face);
    const TensorNURBS<double, d>& basis = patch.basis();

    Eigen::MatrixXd values;
    std::array<Eigen::MatrixXd, d> gradients;
    Eigen::VectorXi actives;
    PointMatrix<double, n> positions;
    PointMatrix<double, n> normals;
    double error = 0.;

    for (const auto& element : basis.grid()) {
        // Points at a different fraction of the element in each direction
        const std::array fractions{.2, .5, .8};
        Eigen::MatrixXd points(fractions.size(), d);

        for (std::size_t direction = 0; direction < d; ++direction) {
            const double start = element.start()[direction];
            const double width = element.end()[direction] - start;

            for (std::size_t point = 0; point < fractions.size(); ++point)
                points(point, direction) =
                    start + fractions[(point + direction) % 3] * width;
        }

        basis.active_on_element(element.index(), actives);
        basis.grad_on_element(element.first_active(), points, values,
                              gradients);
        patch.position_on_element(actives, values, positions);
        boundary.normal_on_element(face, actives, gradients, normals);

        PointMatrix<double, n> radial = positions;

        if constexpr (n == 3)
            radial.col(2).setZero();

        radial.rowwise().normalize();
        error = std::max(error,
                         (normals - boundary.sign(face) * radial)
                             .cwiseAbs().maxCoeff());
    }

    return error;
}

} // namespace

TEST_CASE("Boundary requires one sign per face, each 1 or -1", "[boundary]")
{
    using Curves = Boundary<double, 2>;

    const std::vector<Curves::Face> faces{circle()};

    REQUIRE_THROWS_AS(Curves(faces, std::vector<int>(2, 1)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Curves(faces, std::vector<int>(1, 0)),
                      std::invalid_argument);
}

TEST_CASE("Normals of a circle point out of it, and into it with sign -1",
          "[boundary]")
{
    const Boundary<double, 2> boundary({circle(), circle()}, {1, -1});

    REQUIRE(radial_error(boundary, 0) < 1e-14);
    REQUIRE(radial_error(boundary, 1) < 1e-14);
}

TEST_CASE("Normals of a cylinder point radially out of it", "[boundary]")
{
    const Boundary<double, 3> boundary({half_cylinder()}, {1});

    REQUIRE(radial_error(boundary, 0) < 1e-14);
}

TEST_CASE("Points project onto the closest point of a circle",
          "[boundary]")
{
    // The circle moved far from the origin, where the coordinates are large
    // against the radius
    const Eigen::RowVector2d center(1000., -500.);
    const Boundary<double, 2>::Face around_origin = circle();
    const Boundary<double, 2> boundary(
        {{around_origin.basis(),
          around_origin.coefficients().rowwise() + center}},
        {1});

    // Points inside and outside, in every quadrant
    PointMatrix<double, 2> points(12, 2);
    Eigen::Index row = 0;

    for (const double distance : {.5, 1.7, 3.5})
        for (const double angle : {.4, 2.2, 4., 5.9})
            points.row(row++) =
                center
                + distance * Eigen::RowVector2d(std::cos(angle),
                                                std::sin(angle));

    const iguana::BoundaryProjection<double> projection =
        iguana::project_points(boundary, points);

    for (Eigen::Index point = 0; point < points.rows(); ++point) {
        const Eigen::RowVector2d radial =
            (points.row(point) - center).normalized();

        // The arc of each quadrant spans a quarter of the parameters
        const double angle = std::atan2(radial(1), radial(0));
        const double turn = angle < 0. ? angle + 2. * pi : angle;
        const double quadrant = std::floor(turn / (pi / 2.));

        REQUIRE((projection.positions.row(point) - (center + radius * radial))
                    .norm() < 1e-12);
        REQUIRE((projection.normals.row(point) - radial).norm() < 1e-12);
        REQUIRE(projection.faces(point) == 0);
        REQUIRE(std::floor(4. * projection.parameters(point)) == quadrant);
    }
}

TEST_CASE("Points project onto the sides and corners of a square",
          "[boundary]")
{
    using Curves = Boundary<double, 2>;

    // The unit square, counterclockwise from the origin
    const Curves square(
        {segment({0., 0.}, {1., 0.}), segment({1., 0.}, {1., 1.}),
         segment({1., 1.}, {0., 1.}), segment({0., 1.}, {0., 0.})},
        {1, 1, 1, 1});

    // Off a corner, inside near a side, and outside below another
    PointMatrix<double, 2> points(3, 2);
    points << 1.3, 1.2,
              .4, .9,
              .25, -.5;

    const iguana::BoundaryProjection<double> projection =
        iguana::project_points(square, points);

    // The corner is found first at the end of the right side, whose normal
    // it takes
    PointMatrix<double, 2> positions(3, 2);
    positions << 1., 1.,
                 .4, 1.,
                 .25, 0.;

    PointMatrix<double, 2> normals(3, 2);
    normals << 1., 0.,
               0., 1.,
               0., -1.;

    REQUIRE((projection.positions - positions).cwiseAbs().maxCoeff() < 1e-15);
    REQUIRE((projection.normals - normals).cwiseAbs().maxCoeff() < 1e-15);
    REQUIRE(projection.faces == Eigen::Vector3i(1, 2, 0));

    // A boundary without faces, or with a face whose knots are not clamped,
    // has nothing to project onto
    const TensorBSpline<double, 1> unclamped(
        {BSpline<double>(1, {0., 1., 2., 3.})});
    PointMatrix<double, 2> ends(2, 2);
    ends << 0., 0., 1., 0.;

    const Curves open({{TensorNURBS<double, 1>(unclamped,
                                               Eigen::Vector2d::Ones()),
                        ends}},
                      {1});

    REQUIRE_THROWS_AS(iguana::project_points(Curves({}, {}), points),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::project_points(open, points),
                      std::invalid_argument);
}
