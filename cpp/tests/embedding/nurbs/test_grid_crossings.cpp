/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/embedding/nurbs/grid_crossings.hpp>
#include <iguana/grid/knot_vector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::KnotVector;
using iguana::PointMatrix;
using iguana::TensorBSpline;
using iguana::TensorNURBS;

using Curve = iguana::Patch<TensorNURBS<double, 1>, 2>;
using Piece = iguana::CurvePiece<double>;
using Lines = std::array<std::vector<double>, 2>;

/// @brief Quadratic NURBS curve
Curve curve(std::vector<double> knots, const PointMatrix<double, 2>& points,
            const Eigen::VectorXd& weights)
{
    const TensorBSpline<double, 1> bspline(
        {BSpline<double>(2, std::move(knots))});

    return {TensorNURBS<double, 1>(bspline, weights), points};
}

/// @brief Quadratic NURBS curve of one element
Curve arc(const PointMatrix<double, 2>& points,
          const Eigen::Vector3d& weights)
{
    return curve({0., 0., 0., 1., 1., 1.}, points, weights);
}

/// @brief Pieces into which lines divide an element of a curve
std::vector<Piece> divide(const Curve& curve, int element,
                          const Lines& lines, int sign = 1)
{
    const std::vector<Eigen::MatrixXd> operators =
        iguana::extraction_operators(
            curve.basis().bspline().axis(0).knots());

    return iguana::divide_curve(
        curve, element, operators[static_cast<std::size_t>(element)],
        lines, sign, 4096 * std::numeric_limits<double>::epsilon());
}

/// @brief Point of an element of a curve at a parameter of [0, 1] of the
///        element
Eigen::Vector2d point_at(const Curve& curve, int element, double t)
{
    const TensorNURBS<double, 1>& basis = curve.basis();
    const KnotVector<double>& knots = basis.bspline().axis(0).knots();
    const double start = knots.element_start(element);
    const double end = knots.element_end(element);

    Eigen::MatrixXd parameters(1, 1);
    parameters << start + t * (end - start);

    Eigen::VectorXi actives;
    Eigen::MatrixXd values;
    PointMatrix<double, 2> positions;

    basis.active_on_element(element, actives);
    basis.eval_on_element({basis.bspline().axis(0).first_active(element)},
                          parameters, values);
    curve.position_on_element(actives, values, positions);

    return positions.row(0).transpose();
}

/**
 * @brief Checks that pieces divide an element of a curve over the cells of
 *        a grid
 *
 * The pieces increase without overlapping in [0, 1], points inside a piece
 * lie in its cell, an end of a piece other than 0 and 1 lies on a line,
 * and pieces meeting at a parameter lie in cells sharing an edge or a
 * vertex
 */
void check_pieces(const Curve& curve, int element, const Lines& lines,
                  const std::vector<Piece>& pieces)
{
    constexpr double tolerance = 1e-12;
    const int columns = static_cast<int>(lines[0].size()) - 1;

    for (std::size_t entry = 0; entry < pieces.size(); ++entry) {
        const Piece& piece = pieces[entry];
        const std::array<int, 2> cell{piece.cell % columns,
                                      piece.cell / columns};

        REQUIRE(0. <= piece.start);
        REQUIRE(piece.start < piece.end);
        REQUIRE(piece.end <= 1.);

        for (const double ratio : {.1, .5, .9}) {
            const Eigen::Vector2d point = point_at(
                curve, element,
                piece.start + ratio * (piece.end - piece.start));

            for (std::size_t axis = 0; axis < 2; ++axis) {
                const std::size_t index = static_cast<std::size_t>(cell[axis]);

                REQUIRE(point(axis) > lines[axis][index] - tolerance);
                REQUIRE(point(axis) < lines[axis][index + 1] + tolerance);
            }
        }

        for (const double t : {piece.start, piece.end}) {
            if (t == 0. || t == 1.)
                continue;

            const Eigen::Vector2d point = point_at(curve, element, t);
            double distance = std::numeric_limits<double>::infinity();

            for (std::size_t axis = 0; axis < 2; ++axis)
                for (const double line : lines[axis])
                    distance = std::min(distance, std::abs(point(axis) - line));

            REQUIRE(distance < tolerance);
        }

        if (entry == 0)
            continue;

        const Piece& before = pieces[entry - 1];
        REQUIRE(before.end <= piece.start);

        if (before.end == piece.start) {
            REQUIRE(std::abs(before.cell % columns - cell[0]) <= 1);
            REQUIRE(std::abs(before.cell / columns - cell[1]) <= 1);
        }
    }
}

} // namespace

TEST_CASE("The elements of a circle split into pieces over the cells it "
          "crosses", "[grid_crossings]")
{
    const double corner = std::sqrt(.5);

    Eigen::VectorXd weights(9);
    weights << 1., corner, 1., corner, 1., corner, 1., corner, 1.;

    // Radius 2 around the origin, counterclockwise from (2, 0), as four
    // quadratic arcs joined at double knots
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

    const Curve circle =
        curve({0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.}, points,
              weights);
    const Lines lines{std::vector{-3., -1., 1., 3.},
                      std::vector{-3., -1., 1., 3.}};

    // One piece in each corner cell and two in each edge cell, none in the
    // middle one, the pieces of each element covering it
    std::array<int, 9> counts{};

    for (int element = 0; element < 4; ++element) {
        const std::vector<Piece> pieces = divide(circle, element, lines);
        check_pieces(circle, element, lines, pieces);

        double covered = 0.;

        for (const Piece& piece : pieces) {
            ++counts[static_cast<std::size_t>(piece.cell)];
            covered += piece.end - piece.start;
        }

        REQUIRE_THAT(covered, WithinAbs(1., 1e-15));
    }

    REQUIRE(counts == std::array{1, 2, 1, 2, 0, 2, 1, 2, 1});
}

TEST_CASE("Pieces end at a tip and leave out the parts outside the grid",
          "[grid_crossings]")
{
    // From a tip at (1/2, 1/2) to (5/2, 1/2), beyond the grid, with
    // x = 1/2 + 2t, crossing x = 1 at t = 1/4 and x = 2 at t = 3/4
    PointMatrix<double, 2> points(3, 2);
    points << .5, .5,
              1.5, .75,
              2.5, .5;

    const Curve open = arc(points, Eigen::Vector3d::Ones());
    const Lines lines{std::vector{0., 1., 2.}, std::vector{0., 1.}};
    const std::vector<Piece> pieces = divide(open, 0, lines);

    check_pieces(open, 0, lines, pieces);

    REQUIRE(pieces.size() == 2);
    REQUIRE(pieces[0].cell == 0);
    REQUIRE(pieces[1].cell == 1);
    REQUIRE(pieces[0].start == 0.);
    REQUIRE_THAT(pieces[0].end, WithinAbs(.25, 1e-14));
    REQUIRE(pieces[1].start == pieces[0].end);
    REQUIRE_THAT(pieces[1].end, WithinAbs(.75, 1e-14));
}

TEST_CASE("An element whose control points coincide has no pieces",
          "[grid_crossings]")
{
    const PointMatrix<double, 2> points =
        PointMatrix<double, 2>::Constant(3, 2, .25);
    const Lines lines{std::vector{0., 1.}, std::vector{0., 1.}};

    REQUIRE(divide(arc(points, Eigen::Vector3d::Ones()), 0, lines).empty());
}

TEST_CASE("Crossings of both lines at a vertex of the grid merge",
          "[grid_crossings]")
{
    // Along y = 3x - 3/5 through the vertex (3/10, 3/10), unevenly
    // parametrized, where the crossings of x = 3/10 and y = 3/10 fall a
    // rounding apart
    PointMatrix<double, 2> points(3, 2);
    points << .2, 0.,
              .3, .3,
              .4, .6;

    const Curve line = arc(points, Eigen::Vector3d(1., 2., .4));
    const Lines lines{std::vector{0., .3, 1.}, std::vector{0., .3, 1.}};
    const std::vector<Piece> pieces = divide(line, 0, lines);

    check_pieces(line, 0, lines, pieces);

    REQUIRE(pieces.size() == 2);
    REQUIRE(pieces[0].cell == 0);
    REQUIRE(pieces[1].cell == 3);
}

TEST_CASE("A piece touching a line stays in its cell", "[grid_crossings]")
{
    // A parabola rising from y = 1/2 to touch y = 1 at its middle, t = 1/2,
    // without crossing it
    PointMatrix<double, 2> points(3, 2);
    points << 0., .5,
              .5, 1.5,
              1., .5;

    const Curve parabola = arc(points, Eigen::Vector3d::Ones());
    const Lines lines{std::vector{0., 1.}, std::vector{0., 1., 2.}};
    const std::vector<Piece> pieces = divide(parabola, 0, lines);

    check_pieces(parabola, 0, lines, pieces);

    REQUIRE(pieces.size() == 1);
    REQUIRE(pieces[0].cell == 0);
}

TEST_CASE("An element on a line goes to the cells opposite its normal",
          "[grid_crossings]")
{
    // Up along x = 1/2, whose normal points to +x with sign 1 and to -x
    // with sign -1, so that the cells on the left, 0 and 2, or those on the
    // right, 1 and 3, hold it
    const auto [sign, lower, upper] =
        GENERATE(table<int, int, int>({{1, 0, 2}, {-1, 1, 3}}));

    PointMatrix<double, 2> points(3, 2);
    points << .5, .25,
              .5, .5,
              .5, .75;

    const Curve line = arc(points, Eigen::Vector3d(1., 2., 1.));
    const Lines lines{std::vector{0., .5, 1.}, std::vector{0., .5, 1.}};
    const std::vector<Piece> pieces = divide(line, 0, lines, sign);

    check_pieces(line, 0, lines, pieces);

    REQUIRE(pieces.size() == 2);
    REQUIRE(pieces[0].cell == lower);
    REQUIRE(pieces[1].cell == upper);
}
