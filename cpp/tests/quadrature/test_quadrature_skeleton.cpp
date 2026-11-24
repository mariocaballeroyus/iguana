/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::KnotVector;
using iguana::TensorGrid;

using Gauss = iguana::GaussLegendre<double, 1>;
using Quadrature = iguana::QuadratureSkeleton<double, 2>;

} // namespace

TEST_CASE("A skeleton quadrature places a rule on every face of its list",
          "[quadrature_skeleton]")
{
    // Three by two elements over uneven spans, whose seven interior faces
    // are the four normal to the first direction, then the three normal to
    // the second
    const TensorGrid<double, 2> grid(
        {KnotVector<double>(1, {0., 0., .2, .6, 1., 1.}),
         KnotVector<double>(1, {0., 0., .3, 1., 1.})});
    const std::vector<int> faces{6, 0, 3};
    const Quadrature quadrature(grid, faces, Gauss(3));

    REQUIRE(quadrature.num_faces() == 3);
    REQUIRE(quadrature.num_points() == 9);

    // In the order of the list, each face carries the rule across its
    // direction, with weights adding up to its length and the unit normal
    // from the cell before it to the cell after it
    for (int index = 0; index < quadrature.num_faces(); ++index) {
        const int face = faces[static_cast<std::size_t>(index)];
        const std::size_t direction = grid.face_direction(face);
        const std::size_t across = 1 - direction;
        const std::array<double, 2> start = grid.face_start(face);
        const std::array<double, 2> end = grid.face_end(face);
        const std::array<int, 2> cells = grid.face_elements(face);
        const int first = quadrature.offsets()(index);

        REQUIRE(quadrature.offsets()(index + 1) - first == 3);
        REQUIRE(quadrature.directions()(index) == static_cast<int>(direction));
        REQUIRE(quadrature.cells()(index, 0) == cells[0]);
        REQUIRE(quadrature.cells()(index, 1) == cells[1]);

        double length = 0.;

        for (int point = first; point < first + 3; ++point) {
            REQUIRE(quadrature.points()(point, direction) == start[direction]);
            REQUIRE(quadrature.points()(point, across) > start[across]);
            REQUIRE(quadrature.points()(point, across) < end[across]);
            REQUIRE(quadrature.normals()(point, direction) == 1.);
            REQUIRE(quadrature.normals()(point, across) == 0.);

            length += quadrature.weights()(point);
        }

        REQUIRE_THAT(length, WithinAbs(end[across] - start[across], 1e-15));
    }

    // Indices that number no interior face of the grid
    REQUIRE_THROWS_AS(Quadrature(grid, std::vector<int>{-1}, Gauss(3)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Quadrature(grid, std::vector<int>{7}, Gauss(3)),
                      std::invalid_argument);
}

TEST_CASE("A skeleton quadrature of a hierarchical grid spans each face",
          "[quadrature_skeleton]")
{
    // The middle element of three by three refined, and one of its children
    // again, so that faces join elements up to two levels apart
    const iguana::HierarchicalGrid<double, 2> coarse(TensorGrid<double, 2>(
        {KnotVector<double>(1, {0., 0., .2, .6, 1., 1.}),
         KnotVector<double>(1, {0., 0., .3, .5, 1., 1.})}));
    const iguana::HierarchicalGrid<double, 2> once =
        iguana::refine(coarse, std::vector<int>{4});
    const iguana::HierarchicalGrid<double, 2> grid =
        iguana::refine(once, std::vector<int>{8});

    std::vector<int> faces(static_cast<std::size_t>(grid.num_faces()));

    for (int face = 0; face < grid.num_faces(); ++face)
        faces[static_cast<std::size_t>(face)] = face;

    const Quadrature quadrature(grid, faces, Gauss(2));

    REQUIRE(quadrature.num_faces() == grid.num_faces());

    // Each face carries its elements, and its rule lies on it and measures
    // its length
    for (int face = 0; face < grid.num_faces(); ++face) {
        const std::size_t direction = grid.face_direction(face);
        const std::size_t across = 1 - direction;
        const std::array<double, 2> start = grid.face_start(face);
        const std::array<double, 2> end = grid.face_end(face);
        const std::array<int, 2> elements = grid.face_elements(face);
        double length = 0.;

        REQUIRE(quadrature.cells()(face, 0) == elements[0]);
        REQUIRE(quadrature.cells()(face, 1) == elements[1]);

        for (int point = quadrature.offsets()(face);
             point < quadrature.offsets()(face + 1); ++point) {
            const auto along = static_cast<Eigen::Index>(across);
            const auto normal = static_cast<Eigen::Index>(direction);

            REQUIRE(quadrature.points()(point, normal) == start[direction]);
            REQUIRE(quadrature.points()(point, along) > start[across]);
            REQUIRE(quadrature.points()(point, along) < end[across]);
            length += quadrature.weights()(point);
        }

        REQUIRE_THAT(length, WithinAbs(end[across] - start[across], 1e-15));
    }
}
