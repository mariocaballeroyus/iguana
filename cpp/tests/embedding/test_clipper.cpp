/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/embedding/clipper.hpp>
#include <iguana/embedding/slicer.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;

/// @brief Triangles of a closed surface, counterclockwise seen from outside
template<std::size_t num_vertices, std::size_t num_triangles>
std::vector<Eigen::Matrix3d>
surface(const std::array<Eigen::Vector3d, num_vertices>& vertices,
        const std::array<std::array<int, 3>, num_triangles>& triangles)
{
    std::vector<Eigen::Matrix3d> result;

    for (const auto& [first, second, third] : triangles) {
        Eigen::Matrix3d triangle;
        triangle << vertices[first], vertices[second], vertices[third];
        result.push_back(triangle);
    }

    return result;
}

/// @brief Unit tetrahedron with a corner at the origin
std::vector<Eigen::Matrix3d> tetrahedron()
{
    return surface<4, 4>({Eigen::Vector3d(0., 0., 0.),
                          Eigen::Vector3d(1., 0., 0.),
                          Eigen::Vector3d(0., 1., 0.),
                          Eigen::Vector3d(0., 0., 1.)},
                         {{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}});
}

/// @brief Unit cube with a corner at the origin, whose vertex c has the
///        bits of c as coordinates
std::vector<Eigen::Matrix3d> cube()
{
    std::array<Eigen::Vector3d, 8> vertices;

    for (int corner = 0; corner < 8; ++corner)
        vertices[corner] = Eigen::Vector3d(corner & 1, (corner >> 1) & 1,
                                           (corner >> 2) & 1);

    return surface<8, 12>(vertices,
                          {{{0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6},
                            {0, 1, 5}, {0, 5, 4}, {2, 6, 7}, {2, 7, 3},
                            {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}}});
}

// The measures below follow the divergence theorem as moment fitting will
// use it, with the field (x_0 - lower_0, 0, ...): the facets clipped to the
// box, plus the extent of the box times the measure of the section at its
// upper face, one dimension down

/// @brief Length inside [lower, upper] of a line crossed at signed points
double length(const std::vector<std::pair<double, int>>& points, double lower,
              double upper)
{
    double result = 0.;

    // Entries minus exits up to the upper end, which tell if it is inside
    int depth = 0;

    for (const auto& [coordinate, sign] : points) {
        if (coordinate > upper)
            continue;

        depth -= sign;

        if (coordinate >= lower)
            result += (coordinate - lower) * sign;
    }

    return result + (depth > 0 ? upper - lower : 0.);
}

/// @brief Area inside the box [lower, upper] of a region bounded by segments
double area(const std::vector<Eigen::Matrix2d>& segments,
            const std::array<double, 2>& lower,
            const std::array<double, 2>& upper)
{
    double result = 0.;
    std::vector<std::pair<double, int>> points;

    for (const Eigen::Matrix2d& segment : segments) {
        // The integrand is linear and n_x ds = dy along the part
        if (const auto part = iguana::clip_segment(segment, lower, upper))
            result += (part->row(0).mean() - lower[0])
                      * ((*part)(1, 1) - (*part)(1, 0));

        if (const auto point = iguana::slice_segment(segment, 0, upper[0]))
            points.push_back(*point);
    }

    return result + (upper[0] - lower[0]) * length(points, lower[1], upper[1]);
}

/// @brief Volume inside the box [lower, upper] of a solid bounded by
///        triangles
double volume(const std::vector<Eigen::Matrix3d>& triangles,
              const std::array<double, 3>& lower,
              const std::array<double, 3>& upper)
{
    double result = 0.;
    std::vector<Eigen::Matrix2d> segments;

    for (const Eigen::Matrix3d& triangle : triangles) {
        const auto polygon = iguana::clip_triangle(triangle, lower, upper);

        // Triangles from the first vertex, where the integrand is linear
        // and n_x dS is half the first component of the cross product
        for (Eigen::Index vertex = 1; vertex + 1 < polygon.cols(); ++vertex) {
            const Eigen::Vector3d first = polygon.col(0);
            const Eigen::Vector3d second = polygon.col(vertex);
            const Eigen::Vector3d third = polygon.col(vertex + 1);

            result += ((first(0) + second(0) + third(0)) / 3. - lower[0])
                      * (second - first).cross(third - first)(0) / 2.;
        }

        if (const auto segment = iguana::slice_triangle(triangle, 0, upper[0]))
            segments.push_back(*segment);
    }

    return result + (upper[0] - lower[0])
                        * area(segments, {lower[1], lower[2]},
                               {upper[1], upper[2]});
}

/**
 * @brief Volume of the unit tetrahedron inside the box [lower, upper]
 *
 * By inclusion and exclusion over the corners c of the box, from the
 * volumes (1 - c_0 - c_1 - c_2)^3 / 6 of the tetrahedron beyond them
 */
double tetrahedron_volume(const std::array<double, 3>& lower,
                          const std::array<double, 3>& upper)
{
    double result = 0.;

    for (int corner = 0; corner < 8; ++corner) {
        double sum = 0.;
        double sign = 1.;

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const bool is_upper = (corner >> axis) & 1;

            sum += std::max(is_upper ? upper[axis] : lower[axis], 0.);
            sign *= is_upper ? -1. : 1.;
        }

        const double side = std::max(1. - sum, 0.);
        result += sign * side * side * side / 6.;
    }

    return result;
}

} // namespace

TEST_CASE("Clipped facets and the upper section give the volume in a box",
          "[embedding]")
{
    // Slanted facets clipped through the box, then the tetrahedron's faces
    // on the lower faces of the box
    for (const auto& [lower, upper] :
         {std::pair{std::array{0.2, 0.1, 0.}, std::array{0.6, 0.5, 0.3}},
          std::pair{std::array{0., 0., 0.}, std::array{0.5, 0.5, 0.5}}})
        REQUIRE_THAT(volume(tetrahedron(), lower, upper),
                     WithinAbs(tetrahedron_volume(lower, upper), 1e-14));

    // Facets on the upper faces of the box, with the solid inside the box
    // and outside it
    REQUIRE_THAT(volume(cube(), {0.5, 0.5, 0.5}, {1., 1., 1.}),
                 WithinAbs(0.125, 1e-14));
    REQUIRE_THAT(volume(tetrahedron(), {-0.5, 0., 0.}, {0., 1., 1.}),
                 WithinAbs(0., 1e-14));
}

TEST_CASE("Clipped segments and the upper section give the area in a box",
          "[embedding]")
{
    // Triangle x, y >= 0 with x + y <= 1, counterclockwise
    Eigen::Matrix2d first;
    Eigen::Matrix2d second;
    Eigen::Matrix2d third;
    first << 0., 1., 0., 0.;
    second << 1., 0., 0., 1.;
    third << 0., 0., 1., 0.;

    // The box [0.2, 0.6] x [0.1, 0.5] less its corner beyond x + y = 1
    REQUIRE_THAT(area({first, second, third}, {0.2, 0.1}, {0.6, 0.5}),
                 WithinAbs(0.4 * 0.4 - 0.1 * 0.1 / 2., 1e-14));
}
