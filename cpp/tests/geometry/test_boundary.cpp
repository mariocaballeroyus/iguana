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
#include <tuple>
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

/// @brief Patch of one bilinear element stretching the unit square onto a
///        box of the plane, from its lowest corner and with its sizes
iguana::Patch<TensorBSpline<double, 2>, 2> box(
    const Eigen::RowVector2d& corner, const Eigen::RowVector2d& sizes)
{
    const BSpline<double> unit(1, {0., 0., 1., 1.});
    const TensorBSpline<double, 2> basis({unit, unit});

    // The corners, with the first direction running fastest
    PointMatrix<double, 2> corners(4, 2);
    corners << 0., 0.,
               1., 0.,
               0., 1.,
               1., 1.;
    corners = corners.array().rowwise() * sizes.array();
    corners.rowwise() += corner;

    return {basis, corners};
}

/// @brief Patch of one bilinear element mapped onto itself, so that its
///        parameters are positions in the plane
iguana::Patch<TensorBSpline<double, 2>, 2> identity()
{
    return box({0., 0.}, {1., 1.});
}

/// @brief Patch of bilinear elements of equal size covering the square
///        [−3, 3]^2, a background whose cells size the data along a
///        boundary
iguana::Patch<TensorBSpline<double, 2>, 2> cells(int count)
{
    std::vector<double> knots{0.};

    for (int knot = 0; knot <= count; ++knot)
        knots.push_back(static_cast<double>(knot) / count);

    knots.push_back(1.);

    const BSpline<double> axis(1, knots);
    PointMatrix<double, 2> nodes((count + 1) * (count + 1), 2);

    for (int j = 0; j <= count; ++j)
        for (int i = 0; i <= count; ++i)
            nodes.row(j * (count + 1) + i) =
                Eigen::RowVector2d(-3. + 6. * i / count, -3. + 6. * j / count);

    return {TensorBSpline<double, 2>({axis, axis}), nodes};
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
        iguana::project_closest_points(identity(), boundary, points);

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
        iguana::project_closest_points(identity(), square, points);

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

    REQUIRE_THROWS_AS(
        iguana::project_closest_points(identity(), Curves({}, {}), points),
        std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::project_closest_points(identity(), open, points),
                      std::invalid_argument);
}

TEST_CASE("The distance vectors of a projection lie in the parameters of "
          "its patch", "[boundary]")
{
    // A patch stretching the unit square onto [10, 14] x [5, 6], in the
    // middle of which lies the circle
    const Eigen::RowVector2d corner(10., 5.);
    const Eigen::RowVector2d sizes(4., 1.);
    const iguana::Patch<TensorBSpline<double, 2>, 2> stretched =
        box(corner, sizes);

    const Eigen::RowVector2d center(12., 5.5);
    const Boundary<double, 2>::Face around_origin = circle();
    const Boundary<double, 2> boundary(
        {{around_origin.basis(),
          around_origin.coefficients().rowwise() + center}},
        {1});

    // Points of the parameter space inside and outside the circle, none at
    // its center
    PointMatrix<double, 2> points(4, 2);
    points << .1, .2,
              .35, .8,
              .6, .5,
              .9, .2;

    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(stretched, boundary, points);

    // The closest points lie on the circle along the radii, and the map
    // carries each point moved by its distance vector onto its closest point
    for (Eigen::Index point = 0; point < points.rows(); ++point) {
        const Eigen::RowVector2d position =
            corner + points.row(point).cwiseProduct(sizes);
        const Eigen::RowVector2d radial = (position - center).normalized();
        const Eigen::RowVector2d closest =
            points.row(point) + projection.distances.row(point);

        REQUIRE((projection.positions.row(point) - (center + radius * radial))
                    .norm() < 1e-12);
        REQUIRE((corner + closest.cwiseProduct(sizes)
                 - projection.positions.row(point)).norm() < 1e-12);
    }

    // A map that is not affine, one corner of the box moved away
    PointMatrix<double, 2> bent = stretched.coefficients();
    bent(3, 0) += .5;
    const iguana::Patch<TensorBSpline<double, 2>, 2> curved(
        stretched.basis(), bent);

    REQUIRE_THROWS_AS(iguana::project_closest_points(curved, boundary, points),
                      std::invalid_argument);
}

TEST_CASE("A circle has the inverse of its radius for curvature, constant "
          "along it", "[boundary]")
{
    using Curves = Boundary<double, 2>;

    // Points inside and outside, some on the radii through the double knots
    PointMatrix<double, 2> points(16, 2);
    Eigen::Index row = 0;

    for (const double distance : {1.3, 2.9})
        for (const double angle : {0., .4, pi / 2., 2.2, pi, 4., 1.5 * pi,
                                   5.9})
            points.row(row++) =
                distance * Eigen::RowVector2d(std::cos(angle),
                                              std::sin(angle));

    // The disk, and the plane around it with the normals turned in
    for (const int sign : {1, -1}) {
        INFO("sign " << sign);
        const Curves boundary({circle()}, {sign});
        const iguana::BoundaryProjection<double> projection =
            iguana::project_closest_points(identity(), boundary, points);
        const Eigen::MatrixXd curvatures =
            iguana::curvature_derivs(boundary, projection, 4);

        REQUIRE(curvatures.rows() == points.rows());
        REQUIRE(curvatures.cols() == 5);
        REQUIRE((curvatures.col(0).array() - sign / radius).abs().maxCoeff()
                < 1e-13);
        REQUIRE(curvatures.rightCols(4).cwiseAbs().maxCoeff() < 1e-12);
    }
}

TEST_CASE("The curvature of an ellipse and its derivatives match their "
          "closed forms, whatever way it is traversed", "[boundary]")
{
    using Curves = Boundary<double, 2>;

    // The circle stretched into an ellipse of semi-axes a and b, which is
    // x = a cos θ, y = b sin θ, and the same ellipse traversed clockwise
    const double a = 3.;
    const double b = 1.2;
    const Curves::Face round = circle();

    PointMatrix<double, 2> stretched = round.coefficients();
    stretched.col(0) *= a / radius;
    stretched.col(1) *= b / radius;

    const Curves::Face ellipse(round.basis(), stretched);
    const Curves::Face reversed(
        TensorNURBS<double, 1>(round.basis().bspline(),
                               round.basis().weights().reverse()),
        stretched.colwise().reverse());

    // Points inside and outside, around the ellipse
    PointMatrix<double, 2> points(12, 2);
    Eigen::Index row = 0;

    for (const double scale : {.7, 1.4})
        for (const double angle : {.3, 1.1, 2., 3.3, 4.4, 5.6})
            points.row(row++) = scale * Eigen::RowVector2d(
                                            a * std::cos(angle),
                                            b * std::sin(angle));

    // The ellipse traversed counterclockwise bounds the domain inside it
    // with sign 1 and outside it with sign -1, the reversed one inside it
    // with sign -1
    for (const auto& [face, sign, inside] :
         {std::tuple{ellipse, 1, true}, std::tuple{ellipse, -1, false},
          std::tuple{reversed, -1, true}}) {
        INFO("sign " << sign << ", inside " << inside);
        const Curves boundary({face}, {sign});
        const iguana::BoundaryProjection<double> projection =
            iguana::project_closest_points(identity(), boundary, points);
        const Eigen::MatrixXd curvatures =
            iguana::curvature_derivs(boundary, projection, 2);

        for (Eigen::Index point = 0; point < points.rows(); ++point) {
            INFO("point " << point);
            const Eigen::RowVector2d closest = projection.positions.row(point);
            const double theta = std::atan2(closest(1) / b, closest(0) / a);
            const double sine = std::sin(theta);
            const double cosine = std::cos(theta);

            // With w = |c'(θ)|, the closed forms along the counterclockwise
            // arc length
            const double w = std::sqrt(a * a * sine * sine
                                       + b * b * cosine * cosine);
            const double c = a * b;
            const double e = a * a - b * b;
            const double kappa = c / std::pow(w, 3.);
            const double first =
                -3. * c * e * sine * cosine / std::pow(w, 6.);
            const double second =
                -3. * c * e
                * (std::cos(2. * theta) / std::pow(w, 7.)
                   - 6. * e * sine * sine * cosine * cosine / std::pow(w, 9.));

            // Outside the ellipse its normal turns in, which turns the
            // curvature and the direction of the arc length
            const double turn = inside ? 1. : -1.;

            REQUIRE(std::abs(curvatures(point, 0) - turn * kappa)
                    < 1e-12 * kappa);
            REQUIRE(std::abs(curvatures(point, 1) - first)
                    < 1e-11 * std::abs(c * e));
            REQUIRE(std::abs(curvatures(point, 2) - turn * second)
                    < 1e-10 * std::abs(c * e * e));
        }
    }
}

TEST_CASE("A straight boundary has no curvature", "[boundary]")
{
    const Boundary<double, 2> line({segment({-1., .5}, {2., 1.3})}, {1});

    PointMatrix<double, 2> points(3, 2);
    points << 0., 0.,
              .5, 2.,
              1.5, .7;

    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(identity(), line, points);

    REQUIRE(iguana::curvature_derivs(line, projection, 3).cwiseAbs()
                .maxCoeff() < 1e-15);
    REQUIRE_THROWS_AS(iguana::curvature_derivs(line, projection, -1),
                      std::invalid_argument);
}

TEST_CASE("Data interpolated along a circle converge to the derivatives of "
          "their values in its arc length", "[boundary]")
{
    using Curves = Boundary<double, 2>;

    const int degree = 4;

    // g = x, whose derivatives along the arc length s = R θ are those of
    // R cos θ, with s running backwards where the normals are turned in
    PointMatrix<double, 2> points(8, 2);
    Eigen::Index row = 0;

    for (const double angle : {.3, 1.2, 2.2, 2.9, 3.6, 4.4, 5.1, 6.})
        points.row(row++) =
            2.6 * Eigen::RowVector2d(std::cos(angle), std::sin(angle));

    for (const int sign : {1, -1}) {
        INFO("sign " << sign);
        const Curves boundary({circle()}, {sign});

        // The largest error of each order over the points, relative to the
        // size R^(1 − j) of the derivative, with cells of two sizes
        std::array<Eigen::VectorXd, 2> errors;

        for (const int count : {8, 32}) {
            const iguana::Patch<TensorBSpline<double, 2>, 2> patch =
                cells(count);
            const PointMatrix<double, 2> nodes =
                iguana::boundary_interpolation_points(patch, boundary, degree);

            REQUIRE((nodes.rowwise().norm().array() - radius).abs().maxCoeff()
                    < 1e-13);

            const iguana::BoundaryProjection<double> projection =
                iguana::project_closest_points(patch, boundary, points);
            const Eigen::MatrixXd derivs =
                iguana::boundary_interpolated_derivs(
                    patch, boundary, degree, Eigen::VectorXd(nodes.col(0)),
                    projection, degree);

            REQUIRE(derivs.rows() == points.rows());
            REQUIRE(derivs.cols() == degree + 1);

            Eigen::VectorXd error = Eigen::VectorXd::Zero(degree + 1);

            for (Eigen::Index point = 0; point < points.rows(); ++point) {
                const Eigen::RowVector2d closest =
                    projection.positions.row(point);
                const double angle = std::atan2(closest(1), closest(0));

                for (int j = 0; j <= degree; ++j) {
                    const double exact = std::pow(sign, j)
                                         * std::pow(radius, 1. - j)
                                         * std::cos(angle + j * pi / 2.);
                    error(j) = std::max(error(j),
                                        std::abs(derivs(point, j) - exact)
                                            / std::pow(radius, 1. - j));
                }
            }

            errors[count == 8 ? 0 : 1] = error;
        }

        // The derivative of order j of an interpolant of degree q converges
        // at order q + 1 − j, so a quarter of the cell size divides its error
        // by about 4^(q + 1 − j), here asked by a quarter of that. The
        // highest one is constant on each element and left out
        INFO("errors " << errors[0].transpose() << " and "
                       << errors[1].transpose());
        REQUIRE(errors[1](0) < 1e-7);

        for (int j = 0; j < degree; ++j)
            REQUIRE(errors[0](j) / errors[1](j)
                    > std::pow(4., degree - j) / 4.);
    }
}

TEST_CASE("Data along a boundary need an affine patch, one value per point "
          "and an order within the degree", "[boundary]")
{
    using Curves = Boundary<double, 2>;

    const Curves boundary({circle()}, {1});
    const iguana::Patch<TensorBSpline<double, 2>, 2> patch = cells(4);
    const PointMatrix<double, 2> nodes =
        iguana::boundary_interpolation_points(patch, boundary, 2);
    const Eigen::VectorXd values = Eigen::VectorXd::Zero(nodes.rows());

    PointMatrix<double, 2> points(1, 2);
    points << 2.5, .5;
    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(patch, boundary, points);

    REQUIRE_THROWS_AS(iguana::boundary_interpolation_points(patch, boundary,
                                                            -1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::boundary_interpolated_derivs(
                          patch, boundary, 2, values, projection, 3),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::boundary_interpolated_derivs(
                          patch, boundary, 2, values, projection, -1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::boundary_interpolated_derivs(
                          patch, boundary, 2,
                          Eigen::VectorXd(values.head(values.size() - 1)),
                          projection, 1),
                      std::invalid_argument);

    // A control point moved off the affine image of its Greville point
    PointMatrix<double, 2> moved = patch.coefficients();
    moved(6, 0) += .1;
    const iguana::Patch<TensorBSpline<double, 2>, 2> curved(patch.basis(),
                                                             moved);

    REQUIRE_THROWS_AS(iguana::boundary_interpolation_points(curved, boundary,
                                                            2),
                      std::invalid_argument);
}
