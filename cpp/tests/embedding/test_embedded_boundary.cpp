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
using Region = iguana::Patch<TensorBSpline<double, 2>, 2>;

/// @brief Quadratic NURBS curve
Face curve(std::vector<double> knots, const PointMatrix<double, 2>& points,
           const Eigen::VectorXd& weights)
{
    const TensorBSpline<double, 1> bspline(
        {BSpline<double>(2, std::move(knots))});

    return {TensorNURBS<double, 1>(bspline, weights), points};
}

/// @brief Circle of radius 2 around the origin, counterclockwise from
///        (2, 0), as four quadratic arcs joined at double knots
Boundary<double, 2> circle(std::vector<double> knots)
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

    return {{curve(std::move(knots), points, weights)}, {1}};
}

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

/// @brief Patch with the basis of another and its control points mapped
///        linearly
Region mapped(const Region& patch, const Eigen::Matrix2d& linear)
{
    return {patch.basis(), patch.coefficients() * linear.transpose()};
}

/// @brief Straight face up along x = coordinate, from y = 1/20 to 2/25
Face upward(double coordinate)
{
    PointMatrix<double, 2> points(3, 2);
    points << coordinate, .05,
              coordinate, .065,
              coordinate, .08;

    return curve({0., 0., 0., 1., 1., 1.}, points, Eigen::Vector3d::Ones());
}

} // namespace

TEST_CASE("Pieces of a circle cover it, ending exactly at its knots",
          "[embedded_boundary]")
{
    // Uneven breaks, where the start of the second arc plus its length
    // rounds away from its end
    const std::array<double, 5> breaks{0., .3, .9, .95, 1.};
    const std::vector<double> knots{0., 0., 0., .3, .3, .9, .9, .95, .95,
                                    1., 1., 1.};
    const Region square = patch_of({-3., -1., 1., 3.}, {-3., -1., 1., 3.});
    const TensorGrid<double, 2>& grid = square.basis().grid();
    const Embedded embedded(square, circle(knots));

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

TEST_CASE("Pieces over a hierarchical grid lie in the active elements "
          "covering them", "[embedded_boundary]")
{
    const std::vector<double> knots{0., 0., 0., .25, .25, .5, .5, .75, .75,
                                    1., 1., 1.};
    const Region square = patch_of({-3., -1., 1., 3.}, {-3., -1., 1., 3.});
    const iguana::HierarchicalGrid<double, 2> coarse(square.basis().grid());

    // Without refinement, the pieces are those of the tensor-product grid
    const Embedded tensor(square, circle(knots));
    const Embedded unrefined(iguana::refine(square, coarse), circle(knots));

    REQUIRE(unrefined.num_elements() == tensor.num_elements());

    for (int element = 0; element < tensor.num_elements(); ++element) {
        const std::span<const Piece> expected =
            tensor.pieces_on_element(element);
        const std::span<const Piece> pieces =
            unrefined.pieces_on_element(element);

        REQUIRE(pieces.size() == expected.size());

        for (std::size_t entry = 0; entry < pieces.size(); ++entry) {
            REQUIRE(pieces[entry].face_element
                    == expected[entry].face_element);
            REQUIRE(pieces[entry].start == expected[entry].start);
            REQUIRE(pieces[entry].end == expected[entry].end);
        }
    }

    // The corner and the middle elements refined, then the child of the
    // corner the circle crosses, so that the grid has three levels
    const std::vector<int> marked{0, 4};
    const iguana::HierarchicalGrid<double, 2> once =
        iguana::refine(coarse, marked);
    const std::vector<int> child{iguana::element_holding(
        once, std::array<double, 2>{-1.5, -1.5})};
    const iguana::HierarchicalGrid<double, 2> grid =
        iguana::refine(once, child);
    const Embedded embedded(iguana::refine(square, grid), circle(knots));

    REQUIRE(grid.num_levels() == 3);
    REQUIRE(embedded.num_elements() == grid.num_elements());

    // Every point of the pieces of an element lies in that element, and
    // the pieces cover the circle once
    const iguana::QuadratureBoundary<double, 2> quadrature(
        embedded, iguana::GaussLegendre<double, 1>(8));
    double covered = 0.;

    for (int group = 0; group < quadrature.num_elements(); ++group) {
        const int element = quadrature.elements()(group);
        const std::array<double, 2> start = grid.element_start(element);
        const std::array<double, 2> end = grid.element_end(element);

        for (const Piece& piece : embedded.pieces_on_element(element))
            covered += piece.end - piece.start;

        for (int point = quadrature.offsets()(group);
             point < quadrature.offsets()(group + 1); ++point)
            for (std::size_t axis = 0; axis < 2; ++axis) {
                const double coordinate = quadrature.points()(
                    point, static_cast<Eigen::Index>(axis));

                REQUIRE(coordinate >= start[axis] - 1e-14);
                REQUIRE(coordinate <= end[axis] + 1e-14);
            }
    }

    REQUIRE_THAT(covered, WithinAbs(1., 1e-15));
    REQUIRE_THAT(quadrature.weights().sum(),
                 WithinAbs(4. * std::acos(-1.), 1e-10));
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
    const Embedded embedded(patch_of({0., 1., 2.}, {0., 1.}), boundary);

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
    const Embedded embedded(patch_of({0., .5, 1.}, {0., .5, 1.}),
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

TEST_CASE("A face drawn along a knot line in the plane lies on it exactly",
          "[embedded_boundary]")
{
    // Seven elements over a width of 7/10, so that x = 3/10 is the knot
    // line 3/7, which inverting the map misses by rounding
    std::vector<double> breaks;

    for (int line = 0; line <= 7; ++line)
        breaks.push_back(line / 7.);

    const Region square =
        mapped(patch_of(breaks, breaks), .7 * Eigen::Matrix2d::Identity());
    const Embedded embedded(
        square, Boundary<double, 2>({upward(.3), upward(.3)}, {1, -1}));

    const double knot = square.basis().grid().lines()[0][3];

    REQUIRE((embedded.boundary().face(0).coefficients().col(0).array()
             == knot)
                .all());

    // The normal points to +x with sign 1 and to -x with sign -1, and each
    // face goes to the element opposite it
    REQUIRE(embedded.pieces_on_element(2).size() == 1);
    REQUIRE(embedded.pieces_on_element(2)[0].face == 0);
    REQUIRE(embedded.pieces_on_element(3).size() == 1);
    REQUIRE(embedded.pieces_on_element(3)[0].face == 1);
}

TEST_CASE("A mirrored patch keeps the side of the normal",
          "[embedded_boundary]")
{
    // x runs from 0 to -1 over two elements, so that the element of
    // parameters [1/2, 1] lies at x < -1/2
    const Region mirrored =
        mapped(patch_of({0., .5, 1.}, {0., .5, 1.}),
               Eigen::Vector2d(-1., 1.).asDiagonal().toDenseMatrix());
    const Embedded embedded(mirrored,
                            Boundary<double, 2>({upward(-.5)}, {1}));

    // The normal points to +x, so the face goes to the element at x < -1/2,
    // whose normal in parameters points the other way
    REQUIRE(embedded.boundary().sign(0) == -1);
    REQUIRE(embedded.pieces_on_element(1).size() == 1);
    REQUIRE(embedded.pieces_on_element(0).empty());
}

TEST_CASE("An embedded boundary rejects a patch that is not affine",
          "[embedded_boundary]")
{
    PointMatrix<double, 2> bent = patch_of({0., .5, 1.}, {0., 1.})
                                      .coefficients();
    bent(1, 1) += .1;

    const Region patch(patch_of({0., .5, 1.}, {0., 1.}).basis(), bent);

    // Even without faces to pull back
    REQUIRE_THROWS_AS(Embedded(patch, Boundary<double, 2>({}, {})),
                      std::invalid_argument);
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

    REQUIRE_THROWS_AS(Embedded(patch_of({0., 1.}, {0., 1.}), boundary),
                      std::invalid_argument);
}
