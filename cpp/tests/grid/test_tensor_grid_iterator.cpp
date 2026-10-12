/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cstddef>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::KnotVector;
using iguana::TensorGrid;

/// @brief Grid mixing degrees, a repeated interior knot and an unclamped
///        direction
TensorGrid<double, 3> mixed()
{
    return TensorGrid<double, 3>(
        {KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 3., 3.}),
         KnotVector<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
         KnotVector<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5})});
}

/// @brief Element of each direction, split from the flat index: the
///        reference the walk is checked against
template<std::size_t d>
std::array<int, d> element_of(const TensorGrid<double, d>& grid,
                              int element)
{
    std::array<int, d> indices{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const int count = grid.knots(direction).num_elements();
        indices[direction] = element % count;
        element /= count;
    }

    return indices;
}

} // namespace

TEST_CASE("The walk reaches every element once, in order", "[grid]")
{
    const TensorGrid<double, 3> grid = mixed();

    int visited = 0;
    double volume = 0.;

    for (const auto& element : grid) {
        REQUIRE(element.index() == visited);
        ++visited;

        // The walk advances the element of each direction, so it must
        // agree with splitting the flat index anew
        const std::array<int, 3> indices =
            element_of(grid, element.index());

        double cell = 1.;

        for (std::size_t direction = 0; direction < 3; ++direction) {
            const KnotVector<double>& knots = grid.knots(direction);

            // The first active function is that of a basis on the knots
            REQUIRE(element.first_active()[direction]
                    == BSpline<double>(knots).first_active(
                           indices[direction]));
            REQUIRE(element.start()[direction]
                    == knots.element_start(indices[direction]));
            REQUIRE(element.end()[direction]
                    == knots.element_end(indices[direction]));

            cell *= element.end()[direction] - element.start()[direction];
        }

        volume += cell;
    }

    REQUIRE(visited == grid.num_elements());

    // The element boxes fill the parameter box
    double box = 1.;

    for (std::size_t direction = 0; direction < 3; ++direction)
        box *= grid.knots(direction).domain_end()
               - grid.knots(direction).domain_start();

    REQUIRE_THAT(volume, WithinAbs(box, 1e-12));

    // A second walk starts over
    int again = 0;

    for (const auto& element : grid)
        REQUIRE(element.index() == again++);

    REQUIRE(again == grid.num_elements());
}

TEST_CASE("The lines of a grid bound the elements of the walk", "[grid]")
{
    const TensorGrid<double, 3> grid = mixed();
    const std::array<std::vector<double>, 3> lines = grid.lines();

    // One line more than elements along each direction, with none for the
    // repeated knot or the knots outside the unclamped domain
    for (std::size_t direction = 0; direction < 3; ++direction)
        REQUIRE(static_cast<int>(lines[direction].size())
                == grid.knots(direction).num_elements() + 1);

    for (const auto& element : grid) {
        const std::array<int, 3> indices =
            element_of(grid, element.index());

        for (std::size_t direction = 0; direction < 3; ++direction) {
            const std::size_t index =
                static_cast<std::size_t>(indices[direction]);

            REQUIRE(element.start()[direction] == lines[direction][index]);
            REQUIRE(element.end()[direction] == lines[direction][index + 1]);
        }
    }
}

TEST_CASE("The faces of a grid join each pair of neighboring elements once",
          "[grid]")
{
    const TensorGrid<double, 3> grid = mixed();

    std::vector<std::array<double, 3>> starts;
    std::vector<std::array<double, 3>> ends;

    for (const auto& element : grid) {
        starts.push_back(element.start());
        ends.push_back(element.end());
    }

    // Each pair of elements one apart along a direction, as direction,
    // element before and element after
    std::set<std::array<int, 3>> neighbors;

    for (const auto& element : grid) {
        const std::array<int, 3> indices =
            element_of(grid, element.index());

        for (std::size_t direction = 0; direction < 3; ++direction) {
            if (indices[direction] + 1 == grid.knots(direction).num_elements())
                continue;

            int stride = 1;

            for (std::size_t axis = 0; axis < direction; ++axis)
                stride *= grid.knots(axis).num_elements();

            neighbors.insert({static_cast<int>(direction), element.index(),
                              element.index() + stride});
        }
    }

    std::set<std::array<int, 3>> joined;

    for (int face = 0; face < grid.num_faces(); ++face) {
        const std::size_t direction = grid.face_direction(face);
        const std::array<int, 2> elements = grid.face_elements(face);
        const std::size_t before = static_cast<std::size_t>(elements[0]);
        const std::size_t after = static_cast<std::size_t>(elements[1]);

        joined.insert({static_cast<int>(direction), elements[0], elements[1]});

        // Numbered direction by direction
        if (face > 0)
            REQUIRE(grid.face_direction(face - 1) <= direction);

        // The common side: the element before flattened onto its end along
        // the direction, where the element after starts
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double start = axis == direction ? ends[before][axis]
                                                   : starts[before][axis];

            REQUIRE(grid.face_start(face)[axis] == start);
            REQUIRE(grid.face_end(face)[axis] == ends[before][axis]);
        }

        REQUIRE(starts[after][direction] == ends[before][direction]);
    }

    REQUIRE(grid.num_faces() == static_cast<int>(neighbors.size()));
    REQUIRE(joined == neighbors);

    // Along a direction of a single element, no face is normal to it
    const TensorGrid<double, 2> strip(
        {KnotVector<double>(1, {0., 0., .5, 1., 1.}),
         KnotVector<double>(1, {0., 0., 1., 1.})});

    REQUIRE(strip.num_faces() == 1);
    REQUIRE(strip.face_direction(0) == 0);
}
