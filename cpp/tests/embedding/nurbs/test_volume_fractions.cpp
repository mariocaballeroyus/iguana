/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::Boundary;
using iguana::BSpline;
using iguana::CellType;
using iguana::KnotVector;
using iguana::PointMatrix;
using iguana::TensorBSpline;
using iguana::TensorNURBS;

using Basis = TensorBSpline<double, 2>;
using Face = Boundary<double, 2>::Face;

/// @brief Elements along each direction, and the sides of the rectangle,
///        so that the map stretches the axes unevenly
constexpr int count = 7;
constexpr double width = 3.;
constexpr double height = 2.;

/// @brief Measure of each element of the rectangle
constexpr double cell = width / count * height / count;

/// @brief Rectangle of seven by seven quadratic elements, whose control
///        points are the scaled Greville points, so that the map is affine
iguana::Patch<Basis, 2> rectangle()
{
    std::vector<double> knots{0., 0., 0.};

    for (int element = 1; element < count; ++element)
        knots.push_back(static_cast<double>(element) / count);

    knots.insert(knots.end(), {1., 1., 1.});

    const KnotVector<double> along(2, knots);
    const Basis basis(iguana::TensorGrid<double, 2>({along, along}));
    const int functions = count + 2;
    PointMatrix<double, 2> points(functions * functions, 2);

    // Greville points, the first direction running fastest
    for (int j = 0; j < functions; ++j) {
        for (int i = 0; i < functions; ++i)
            points.row(i + functions * j)
                << width * (knots[i + 1] + knots[i + 2]) / 2.,
                height * (knots[j + 1] + knots[j + 2]) / 2.;
    }

    return {basis, points};
}

/// @brief Circle of radius .71 off the knot lines, as four rational
///        quadratic arcs, counterclockwise or reversed into an obstacle
Boundary<double, 2> circle(bool reversed)
{
    const Eigen::RowVector2d centre(1.43, 1.07);
    const double radius = .71;
    const TensorBSpline<double, 1> arc(
        {BSpline<double>(2, {0., 0., 0., 1., 1., 1.})});
    std::vector<Face> faces;

    for (int quarter = 0; quarter < 4; ++quarter) {
        const double start = quarter * std::numbers::pi / 2.;
        const Eigen::RowVector2d from(std::cos(start), std::sin(start));
        const Eigen::RowVector2d to(-from(1), from(0));

        PointMatrix<double, 2> points(3, 2);
        points << centre + radius * from, centre + radius * (from + to),
            centre + radius * to;

        if (reversed)
            points = points.colwise().reverse().eval();

        faces.push_back({TensorNURBS<double, 1>(
                             arc, Eigen::Vector3d(1., std::sqrt(.5), 1.)),
                         points});
    }

    return {faces, {1, 1, 1, 1}};
}

/// @brief Polygon through corners, counterclockwise, as straight faces
Boundary<double, 2> polygon(const std::vector<Eigen::RowVector2d>& corners)
{
    const TensorBSpline<double, 1> line(
        {BSpline<double>(1, {0., 0., 1., 1.})});
    std::vector<Face> faces;
    std::vector<int> signs;

    for (std::size_t corner = 0; corner < corners.size(); ++corner) {
        PointMatrix<double, 2> points(2, 2);
        points << corners[corner], corners[(corner + 1) % corners.size()];

        faces.push_back(
            {TensorNURBS<double, 1>(line, Eigen::Vector2d::Ones()), points});
        signs.push_back(1);
    }

    return {faces, signs};
}

} // namespace

TEST_CASE("Volume fractions measure a disc and the obstacle it makes",
          "[volume_fractions]")
{
    const iguana::Patch<Basis, 2> patch = rectangle();
    const Eigen::VectorXd disc = iguana::volume_fractions(patch, circle(false));
    const Eigen::VectorXd obstacle =
        iguana::volume_fractions(patch, circle(true));

    // The fractions add up to the area of the disc, and those of the
    // obstacle, the rectangle around it, to the rest
    const double area = std::numbers::pi * .71 * .71;

    REQUIRE_THAT(disc.sum() * cell, WithinAbs(area, 1e-12));
    REQUIRE((disc + obstacle - Eigen::VectorXd::Ones(count * count))
                .cwiseAbs().maxCoeff() < 1e-14);

    // The cells the circle crosses are the cut ones, and the others are
    // exactly inside or outside
    const iguana::EmbeddedBoundary<double, 2> embedded(patch, circle(false));
    const std::vector<CellType> types = iguana::cell_types(disc);

    for (int element = 0; element < count * count; ++element) {
        INFO("element " << element);

        const bool crossed = !embedded.pieces_on_element(element).empty();

        REQUIRE((types[element] == CellType::cut) == crossed);
        REQUIRE(((disc(element) == 0.) || (disc(element) == 1.))
                == !crossed);
    }
}

TEST_CASE("Volume fractions of a polygon with sides on knot lines",
          "[volume_fractions]")
{
    // The bottom and right sides run along the knot lines y = h / 7 and
    // x = 4 w / 7, while the top side slants above y = 4 h / 7 and the left
    // one through the column of cells right of x = w / 7
    const double x = width / count;
    const double y = height / count;
    const std::vector<Eigen::RowVector2d> corners{
        {x, y}, {4. * x, y}, {4. * x, 4. * y}, {x + .3, 4. * y + .1}};

    const Eigen::VectorXd fractions =
        iguana::volume_fractions(rectangle(), polygon(corners));
    double area = 0.;

    for (std::size_t corner = 0; corner < corners.size(); ++corner) {
        const Eigen::RowVector2d& from = corners[corner];
        const Eigen::RowVector2d& to = corners[(corner + 1) % corners.size()];
        area += (from(0) * to(1) - to(0) * from(1)) / 2.;
    }

    REQUIRE_THAT(fractions.sum() * cell, WithinAbs(area, 1e-14));

    // Below the top side, the cells along the sides on knot lines lie
    // inside or outside them, although those sides hold pieces in them and
    // leave rounding in their fractions, and the column of the left side is
    // cut
    const std::vector<CellType> types = iguana::cell_types(fractions);

    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 5; ++column) {
            INFO("cell " << column << ", " << row);

            CellType type = CellType::outside;

            if (row >= 1 && column == 1)
                type = CellType::cut;
            else if (row >= 1 && (column == 2 || column == 3))
                type = CellType::inside;

            REQUIRE(types[column + count * row] == type);
        }
    }
}

TEST_CASE("Volume fractions need an affine patch", "[volume_fractions]")
{
    iguana::Patch<Basis, 2> patch = rectangle();
    PointMatrix<double, 2> bent = patch.coefficients();
    bent(20, 1) += .1;

    REQUIRE_THROWS_AS(iguana::volume_fractions(
                          iguana::Patch<Basis, 2>(patch.basis(), bent),
                          circle(false)),
                      std::invalid_argument);
}
