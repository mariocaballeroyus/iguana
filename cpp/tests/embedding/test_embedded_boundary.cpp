/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
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
using iguana::TensorNURBS;

using Embedded = iguana::EmbeddedBoundary<double, 2>;
using Face = Boundary<double, 2>::Face;
using Piece = Embedded::Piece;

/// @brief Quadratic NURBS curve
Face curve(std::vector<double> knots, const PointMatrix<double, 2>& points,
           const Eigen::VectorXd& weights)
{
    const TensorBSpline<double, 1> bspline(
        {BSpline<double>(2, std::move(knots))});

    return {TensorNURBS<double, 1>(bspline, weights), points};
}

/// @brief Grid of linear elements between breaks in each direction
TensorGrid<double, 2> grid_of(const std::vector<double>& first,
                              const std::vector<double>& second)
{
    const auto knots = [](const std::vector<double>& breaks) {
        std::vector<double> values{breaks.front()};
        values.insert(values.end(), breaks.begin(), breaks.end());
        values.push_back(breaks.back());

        return KnotVector<double>(1, values);
    };

    return TensorGrid<double, 2>({knots(first), knots(second)});
}

} // namespace

TEST_CASE("Pieces of a circle cover it, ending exactly at its knots",
          "[embedded_boundary]")
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

    // Uneven breaks, where the start of the second arc plus its length
    // rounds away from its end
    const std::array<double, 5> breaks{0., .3, .9, .95, 1.};
    const std::vector<double> knots{0., 0., 0., .3, .3, .9, .9, .95, .95,
                                    1., 1., 1.};
    const TensorGrid<double, 2> grid =
        grid_of({-3., -1., 1., 3.}, {-3., -1., 1., 3.});
    const Embedded embedded(
        grid, Boundary<double, 2>({curve(knots, points, weights)}, {1}));

    // One piece in each corner element and two in each edge element, none
    // in the middle one
    const std::array<std::size_t, 9> counts{1, 2, 1, 2, 0, 2, 1, 2, 1};

    // Smallest start and largest end of the pieces of each arc
    std::array<double, 4> starts{1., 1., 1., 1.};
    std::array<double, 4> ends{0., 0., 0., 0.};
    double covered = 0.;

    REQUIRE(embedded.num_elements() == grid.num_elements());
    REQUIRE(embedded.num_pieces() == 12);

    for (int element = 0; element < grid.num_elements(); ++element) {
        const std::span<const Piece> pieces =
            embedded.pieces_on_element(element);

        REQUIRE(pieces.size() == counts[static_cast<std::size_t>(element)]);

        for (const Piece& piece : pieces) {
            const std::size_t arc =
                static_cast<std::size_t>(piece.face_element);

            starts[arc] = std::min(starts[arc], piece.start);
            ends[arc] = std::max(ends[arc], piece.end);
            covered += piece.end - piece.start;
        }
    }

    REQUIRE_THAT(covered, WithinAbs(1., 1e-15));

    for (std::size_t arc = 0; arc < 4; ++arc) {
        REQUIRE(starts[arc] == breaks[arc]);
        REQUIRE(ends[arc] == breaks[arc + 1]);
    }
}

TEST_CASE("Pieces in an element follow their faces, elements and "
          "parameters", "[embedded_boundary]")
{
    // A parabola from (1/2, 1/5) to (1/2, 4/5), bulging past x = 1 between
    // two crossings, and a curve of two elements inside the left element
    PointMatrix<double, 2> bulging(3, 2);
    bulging << .5, .2,
               2., .5,
               .5, .8;

    PointMatrix<double, 2> inside(4, 2);
    inside << .1, .1,
              .2, .3,
              .3, .5,
              .2, .7;

    const Boundary<double, 2> boundary(
        {curve({0., 0., 0., 1., 1., 1.}, bulging, Eigen::Vector3d::Ones()),
         curve({0., 0., 0., .5, 1., 1., 1.}, inside,
               Eigen::Vector4d::Ones())},
        {1, 1});
    const Embedded embedded(grid_of({0., 1., 2.}, {0., 1.}), boundary);

    const std::span<const Piece> left = embedded.pieces_on_element(0);
    const std::span<const Piece> right = embedded.pieces_on_element(1);

    REQUIRE(left.size() == 4);
    REQUIRE(right.size() == 1);

    // The parabola leaves the left element and comes back, in increasing
    // parameter, before the elements of the second face
    const std::array<std::pair<int, int>, 4> order{
        {{0, 0}, {0, 0}, {1, 0}, {1, 1}}};

    for (std::size_t entry = 0; entry < 4; ++entry) {
        REQUIRE(left[entry].face == order[entry].first);
        REQUIRE(left[entry].face_element == order[entry].second);
    }

    REQUIRE(left[0].end == right[0].start);
    REQUIRE(right[0].end == left[1].start);
}

TEST_CASE("A face on a knot line goes to the elements opposite its normal",
          "[embedded_boundary]")
{
    // Up along x = 1/2, whose normal points to +x with sign 1 and to -x
    // with sign -1
    PointMatrix<double, 2> points(3, 2);
    points << .5, .25,
              .5, .5,
              .5, .75;

    const Face line =
        curve({0., 0., 0., 1., 1., 1.}, points, Eigen::Vector3d(1., 2., 1.));
    const Embedded embedded(grid_of({0., .5, 1.}, {0., .5, 1.}),
                            Boundary<double, 2>({line, line}, {1, -1}));

    // Elements 0 and 2 lie left of the line, 1 and 3 right of it
    const std::array<int, 4> faces{0, 1, 0, 1};

    for (int element = 0; element < 4; ++element) {
        const std::span<const Piece> pieces =
            embedded.pieces_on_element(element);

        REQUIRE(pieces.size() == 1);
        REQUIRE(pieces[0].face == faces[static_cast<std::size_t>(element)]);
    }
}

TEST_CASE("An embedded boundary rejects faces that are not clamped",
          "[embedded_boundary]")
{
    PointMatrix<double, 2> points(3, 2);
    points << 0., .5,
              .5, .5,
              1., .5;

    const Boundary<double, 2> boundary(
        {curve({0., .5, 1., 1.5, 2., 2.5}, points, Eigen::Vector3d::Ones())},
        {1});

    REQUIRE_THROWS_AS(Embedded(grid_of({0., 1.}, {0., 1.}), boundary),
                      std::invalid_argument);
}
