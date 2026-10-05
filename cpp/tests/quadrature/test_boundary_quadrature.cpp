/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::Boundary;
using iguana::KnotVector;
using iguana::PointMatrix;
using iguana::TensorBSpline;
using iguana::TensorGrid;
using iguana::TensorGridIterator;
using iguana::TensorNURBS;

using Embedded = iguana::EmbeddedBoundary<double, 2>;
using Face = Boundary<double, 2>::Face;
using Gauss = iguana::GaussLegendre<double, 1>;
using Quadrature = iguana::BoundaryQuadrature<double, 2>;
using Region = iguana::Patch<TensorBSpline<double, 2>, 2>;

/// @brief Patch of linear elements between breaks in each direction,
///        mapped onto itself so that its parameters are its coordinates
Region patch_of(const std::vector<double>& first,
                const std::vector<double>& second)
{
    const auto knots = [](const std::vector<double>& breaks) {
        std::vector<double> values{breaks.front()};
        values.insert(values.end(), breaks.begin(), breaks.end());
        values.push_back(breaks.back());

        return KnotVector<double>(1, values);
    };

    const TensorBSpline<double, 2> basis(
        TensorGrid<double, 2>({knots(first), knots(second)}));

    // The control points of linear functions lie at the breaks
    PointMatrix<double, 2> points(basis.num_functions(), 2);

    for (std::size_t row = 0; row < second.size(); ++row)
        for (std::size_t column = 0; column < first.size(); ++column)
            points.row(static_cast<Eigen::Index>(
                column + first.size() * row)) << first[column], second[row];

    return {basis, points};
}

/// @brief Straight face from one point to another
Face segment(const Eigen::RowVector2d& from, const Eigen::RowVector2d& to)
{
    PointMatrix<double, 2> points(2, 2);
    points << from, to;

    const TensorBSpline<double, 1> bspline(
        {BSpline<double>(1, {0., 0., 1., 1.})});

    return {TensorNURBS<double, 1>(bspline, Eigen::Vector2d::Ones()), points};
}

/// @brief Circle of radius two around a center, counterclockwise, as four
///        quadratic arcs joined at double knots
Face circle(const Eigen::RowVector2d& center)
{
    const double corner = std::sqrt(.5);

    Eigen::VectorXd weights(9);
    weights << 1., corner, 1., corner, 1., corner, 1., corner, 1.;

    PointMatrix<double, 2> points(9, 2);
    points << 2., 0.,
              2., 2.,
              0., 2.,
             -2., 2.,
             -2., 0.,
             -2., -2.,
              0., -2.,
              2., -2.,
              2., 0.;
    points.rowwise() += center;

    const TensorBSpline<double, 1> bspline({BSpline<double>(
        2, {0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.})});

    return {TensorNURBS<double, 1>(bspline, weights), points};
}

} // namespace

TEST_CASE("Weights and normals integrate over a polygon exactly",
          "[boundary_quadrature]")
{
    // A tilted quadrilateral, counterclockwise, one face per side, across
    // uneven elements
    const std::array<Eigen::RowVector2d, 4> vertices{
        Eigen::RowVector2d(-2.5, -2.2), Eigen::RowVector2d(2.6, -1.4),
        Eigen::RowVector2d(1.9, 2.5), Eigen::RowVector2d(-1.7, 1.8)};

    std::vector<Face> faces;
    double area = 0.;

    for (std::size_t side = 0; side < 4; ++side) {
        const Eigen::RowVector2d& from = vertices[side];
        const Eigen::RowVector2d& to = vertices[(side + 1) % 4];

        faces.push_back(segment(from, to));
        area += (from(0) * to(1) - to(0) * from(1)) / 2.;
    }

    const Embedded embedded(patch_of({-3., -1., .5, 3.}, {-3., -1.5, 1., 3.}),
                            Boundary<double, 2>(faces, {1, 1, 1, 1}));
    const Quadrature quadrature(embedded, Gauss(2));

    // The weights of each face add up to its length, and the flux of the
    // position, whose divergence is two, to twice the area
    std::array<double, 4> lengths{};
    double flux = 0.;

    for (int point = 0; point < quadrature.num_points(); ++point) {
        const double weight = quadrature.weights()(point);
        const std::size_t side =
            static_cast<std::size_t>(quadrature.faces()(point));

        lengths[side] += weight;
        flux += weight * quadrature.points().row(point).dot(
                             quadrature.normals().row(point));
    }

    for (std::size_t side = 0; side < 4; ++side)
        REQUIRE_THAT(lengths[side],
                     WithinAbs((vertices[(side + 1) % 4] - vertices[side])
                                   .norm(),
                               1e-14));

    REQUIRE_THAT(flux, WithinAbs(2. * area, 1e-13));
}

TEST_CASE("Points are grouped by the elements holding a piece",
          "[boundary_quadrature]")
{
    // The circle misses the middle element, which lies inside it
    const Region square = patch_of({-3., -1., .5, 3.}, {-3., -1.5, 1., 3.});
    const TensorGrid<double, 2>& grid = square.basis().grid();
    const Embedded embedded(
        square, Boundary<double, 2>({circle({.3, -.2})}, {1}));

    const Gauss rule(3);
    const Quadrature quadrature(embedded, rule);

    // Elements holding a piece, in increasing order
    std::vector<int> held;

    for (int element = 0; element < grid.num_elements(); ++element)
        if (!embedded.pieces_on_element(element).empty())
            held.push_back(element);

    // Corners of each element of the grid
    const std::size_t num_elements =
        static_cast<std::size_t>(grid.num_elements());
    std::vector<std::array<double, 2>> starts(num_elements);
    std::vector<std::array<double, 2>> ends(num_elements);

    for (const TensorGridIterator<double, 2>& element : grid) {
        starts[static_cast<std::size_t>(element.index())] = element.start();
        ends[static_cast<std::size_t>(element.index())] = element.end();
    }

    REQUIRE(held.size() == 8);
    REQUIRE(quadrature.num_elements() == 8);

    // Each element holds the whole rule on each of its pieces, all inside
    // it
    for (int position = 0; position < quadrature.num_elements(); ++position) {
        const int element = quadrature.elements()(position);
        const std::size_t index = static_cast<std::size_t>(element);
        const int first = quadrature.offsets()(position);
        const int count = quadrature.offsets()(position + 1) - first;
        const int num_pieces =
            static_cast<int>(embedded.pieces_on_element(element).size());

        REQUIRE(element == held[static_cast<std::size_t>(position)]);
        REQUIRE(count == num_pieces * rule.num_points());

        for (int point = first; point < first + count; ++point)
            for (std::size_t axis = 0; axis < 2; ++axis) {
                const double coordinate = quadrature.points()(
                    point, static_cast<Eigen::Index>(axis));

                REQUIRE(coordinate >= starts[index][axis]);
                REQUIRE(coordinate <= ends[index][axis]);
            }
    }
}

TEST_CASE("Points of a circle lie on it, with normals pointing away from "
          "its center", "[boundary_quadrature]")
{
    const Eigen::RowVector2d center(.3, -.2);
    const Embedded embedded(
        patch_of({-3., -1., .5, 3.}, {-3., -1.5, 1., 3.}),
        Boundary<double, 2>({circle(center)}, {1}));
    const Quadrature quadrature(embedded, Gauss(8));

    for (int point = 0; point < quadrature.num_points(); ++point) {
        const Eigen::RowVector2d radial =
            (quadrature.points().row(point) - center) / 2.;

        REQUIRE((radial - quadrature.normals().row(point)).norm() < 1e-14);
    }

    // The length element of the arcs varies along them
    REQUIRE_THAT(quadrature.weights().sum(),
                 WithinAbs(4. * std::numbers::pi, 1e-12));
}
