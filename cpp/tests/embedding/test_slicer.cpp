/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/embedding/slicer.hpp>

#include <array>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;

/// @brief Triangles of the unit tetrahedron with a corner at the origin,
///        counterclockwise seen from outside
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

/// @brief Segments where the triangles cross the plane x_axis = coordinate
std::vector<Eigen::Matrix2d> slice(const std::vector<Eigen::Matrix3d>& surface,
                                   std::size_t axis, double coordinate)
{
    std::vector<Eigen::Matrix2d> result;

    for (const Eigen::Matrix3d& triangle : surface)
        if (const auto segment =
                iguana::slice_triangle(triangle, axis, coordinate))
            result.push_back(*segment);

    return result;
}

/// @brief Area the segments enclose, positive with the region on their left
double enclosed_area(const std::vector<Eigen::Matrix2d>& segments)
{
    double result = 0.;

    for (const Eigen::Matrix2d& segment : segments)
        result += (segment(0, 0) * segment(1, 1)
                   - segment(0, 1) * segment(1, 0)) / 2.;

    return result;
}

/// @brief Length the segments enclose along the line x_0 = coordinate, the
///        sum of each crossing times its sign
double chord(const std::vector<Eigen::Matrix2d>& segments, double coordinate)
{
    double result = 0.;

    for (const Eigen::Matrix2d& segment : segments)
        if (const auto point = iguana::slice_segment(segment, 0, coordinate))
            result += point->first * point->second;

    return result;
}

} // namespace

TEST_CASE("Slices of a closed surface enclose its cross-sections",
          "[embedding]")
{
    // On every axis, the section at 0.3 is the triangle u, v >= 0 with
    // u + v <= 0.7, and its chord at u = 0.2 runs from v = 0 to 0.5
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::vector<Eigen::Matrix2d> segments =
            slice(tetrahedron(), axis, 0.3);

        REQUIRE_THAT(enclosed_area(segments), WithinAbs(0.7 * 0.7 / 2., 1e-14));
        REQUIRE_THAT(chord(segments, 0.2), WithinAbs(0.5, 1e-14));
    }
}

TEST_CASE("A vertex on the plane counts as below it", "[embedding]")
{
    // The base lies on the plane z = 0 and one leg of its section on the
    // line u = 0, so both slice as if just above
    const std::vector<Eigen::Matrix2d> base = slice(tetrahedron(), 2, 0.);

    REQUIRE_THAT(enclosed_area(base), WithinAbs(0.5, 1e-14));
    REQUIRE_THAT(chord(base, 0.), WithinAbs(1., 1e-14));

    // Only the apex touches the plane z = 1, below it
    REQUIRE(slice(tetrahedron(), 2, 1.).empty());
}
