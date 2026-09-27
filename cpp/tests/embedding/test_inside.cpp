/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/embedding/inside.hpp>

#include <array>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{

using Mask = Eigen::Array<bool, Eigen::Dynamic, 1>;

/// @brief Triangles of the unit tetrahedron, counterclockwise seen from
///        outside
std::vector<Eigen::Matrix3d> tetrahedron()
{
    const std::array<Eigen::Vector3d, 4> vertices{
        Eigen::Vector3d(0., 0., 0.), Eigen::Vector3d(1., 0., 0.),
        Eigen::Vector3d(0., 1., 0.), Eigen::Vector3d(0., 0., 1.)};
    const std::array<std::array<int, 3>, 4> triangles{
        {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}};

    std::vector<Eigen::Matrix3d> result;

    for (const auto& [first, second, third] : triangles) {
        Eigen::Matrix3d triangle;
        triangle << vertices[first], vertices[second], vertices[third];
        result.push_back(triangle);
    }

    return result;
}

/// @brief Triangles of the box [lower, upper], counterclockwise seen from
///        outside, appended to others
void add_box(const Eigen::Vector3d& lower, const Eigen::Vector3d& upper,
             std::vector<Eigen::Matrix3d>& triangles)
{
    // Corner c takes the upper bound in the directions of the bits of c
    std::array<Eigen::Vector3d, 8> corners;

    for (int corner = 0; corner < 8; ++corner)
        for (int axis = 0; axis < 3; ++axis)
            corners[corner](axis) =
                (corner >> axis) & 1 ? upper(axis) : lower(axis);

    const std::array<std::array<int, 3>, 12> faces{
        {{0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
         {2, 6, 7}, {2, 7, 3}, {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}}};

    for (const auto& [first, second, third] : faces) {
        Eigen::Matrix3d triangle;
        triangle << corners[first], corners[second], corners[third];
        triangles.push_back(triangle);
    }
}

/// @brief Points along x_1 at the given coordinates, through (y, z)
Eigen::MatrixXd line(const std::vector<double>& coordinates, double y,
                     double z)
{
    Eigen::MatrixXd result(coordinates.size(), 3);

    for (std::size_t point = 0; point < coordinates.size(); ++point)
        result.row(point) << coordinates[point], y, z;

    return result;
}

} // namespace

TEST_CASE("Points on a line are inside where the domain is", "[embedding]")
{
    // The line (y, z) = (0.2, 0.3) crosses the tetrahedron for 0 < x < 0.5,
    // and the points come in no particular order along it
    const Eigen::MatrixXd points = line({0.6, -0.5, 0.4, 1.5, 0.1}, 0.2, 0.3);
    const Mask expected =
        (Mask(5) << false, false, true, false, true).finished();

    REQUIRE((iguana::is_inside<double, 3>(tetrahedron(), points)
             == expected).all());

    // The single test agrees with the batched one
    for (Eigen::Index point = 0; point < points.rows(); ++point) {
        const Eigen::Vector3d position = points.row(point).transpose();

        REQUIRE(iguana::is_inside(
                    iguana::line_crossings<double, 3>(tetrahedron(), position),
                    position(0))
                == expected(point));
    }

    // In two directions, the triangle x, y >= 0, x + y <= 1 at y = 0.25
    Eigen::Matrix2d first;
    Eigen::Matrix2d second;
    Eigen::Matrix2d third;
    first << 0., 1., 0., 0.;
    second << 1., 0., 0., 1.;
    third << 0., 0., 1., 0.;

    Eigen::MatrixXd planar(3, 2);
    planar << -0.1, 0.25, 0.5, 0.25, 0.8, 0.25;

    const Mask inside =
        iguana::is_inside<double, 2>({first, second, third}, planar);

    REQUIRE((inside == (Mask(3) << false, true, false).finished()).all());
}

TEST_CASE("A point on the boundary counts as just past it along x_1",
          "[embedding]")
{
    std::vector<Eigen::Matrix3d> cube;
    add_box({0., 0., 0.}, {1., 1., 1.}, cube);

    // Entering at x = 0 and leaving at x = 1 both count as before the point
    const Mask inside =
        iguana::is_inside<double, 3>(cube, line({0., 1.}, 0.5, 0.5));

    REQUIRE((inside == (Mask(2) << true, false).finished()).all());
}

TEST_CASE("Overlapping solids are inside where either is", "[embedding]")
{
    std::vector<Eigen::Matrix3d> solids;
    add_box({0., 0., 0.}, {2., 1., 1.}, solids);
    add_box({1., 0., 0.}, {3., 1., 1.}, solids);

    // Entries minus exits count the overlap twice, where parity would not
    const Mask inside = iguana::is_inside<double, 3>(
        solids, line({0.5, 1.5, 2.5, 3.5}, 0.5, 0.5));

    REQUIRE((inside == (Mask(4) << true, true, true, false).finished()).all());

    // The single test counts the same way
    const Eigen::Vector3d overlap(1.5, 0.5, 0.5);

    REQUIRE(iguana::is_inside(
        iguana::line_crossings<double, 3>(solids, overlap), overlap(0)));
}
