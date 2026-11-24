/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::HierarchicalGrid;
using iguana::KnotVector;
using iguana::TensorGrid;

/// @brief Parameters at which an element starts, then those at which it
///        ends
using Box = std::array<std::array<double, 2>, 2>;

} // namespace

TEST_CASE("Refined elements tile the parametric domain", "[grid]")
{
    // Six elements over [0, 4] x [0, 2], with a repeated interior knot and
    // uneven spans
    const HierarchicalGrid<double, 2> coarse(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., 1., 1., 3., 4., 4., 4.}),
         KnotVector<double>(1, {0., 0., .5, 2., 2.})}));

    // Each distinct mark trades an element for its four children, the
    // first ones on a new level
    const HierarchicalGrid<double, 2> once =
        refine(coarse, std::vector<int>{0, 4, 4});

    REQUIRE(once.num_levels() == 2);
    REQUIRE(once.num_elements() == 6 + 2 * 3);

    // Every coarse element and a child, which leaves level 0 empty
    const HierarchicalGrid<double, 2> grid =
        refine(once, std::vector<int>{0, 1, 2, 3, 4});

    REQUIRE(grid.num_levels() == 3);
    REQUIRE(grid.active_elements(0).empty());
    REQUIRE(grid.num_elements() == 12 + 5 * 3);

    std::vector<Box> boxes;
    int level = 0;
    int position = 0;

    for (const auto& element : grid) {
        REQUIRE(element.index() == static_cast<int>(boxes.size()));
        REQUIRE(element.level() >= level);

        if (element.level() > level) {
            level = element.level();
            position = 0;
        }

        // The element of the level grid it stands for, whose walk is the
        // reference
        const int active = grid.active_elements(level)[position];
        ++position;

        for (const auto& reference : grid.level(level)) {
            if (reference.index() != active)
                continue;

            REQUIRE(element.first_active() == reference.first_active());
            REQUIRE(element.start() == reference.start());
            REQUIRE(element.end() == reference.end());
        }

        boxes.push_back({element.start(), element.end()});
    }

    REQUIRE(static_cast<int>(boxes.size()) == grid.num_elements());

    // Pairwise disjoint boxes whose areas add up to that of the parametric
    // domain
    double area = 0.;

    for (std::size_t box = 0; box < boxes.size(); ++box) {
        const auto& [start, end] = boxes[box];
        area += (end[0] - start[0]) * (end[1] - start[1]);

        for (std::size_t other = box + 1; other < boxes.size(); ++other) {
            double overlap = 1.;

            for (std::size_t direction = 0; direction < 2; ++direction)
                overlap *= std::max(
                    0., std::min(end[direction], boxes[other][1][direction])
                            - std::max(start[direction],
                                       boxes[other][0][direction]));

            REQUIRE(overlap == 0.);
        }
    }

    REQUIRE_THAT(area, WithinAbs(4. * 2., 1e-12));
}

TEST_CASE("Active elements are found by index, by point and by the cells "
          "they cover", "[grid]")
{
    const HierarchicalGrid<double, 2> coarse(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., 1., 1., 3., 4., 4., 4.}),
         KnotVector<double>(1, {0., 0., .5, 2., 2.})}));
    const HierarchicalGrid<double, 2> once =
        refine(coarse, std::vector<int>{0, 4});
    const HierarchicalGrid<double, 2> grid =
        refine(once, std::vector<int>{6, 7});

    // Each level starts where the one before ends
    REQUIRE(grid.offset(0) == 0);
    REQUIRE(grid.offset(1) == 4);
    REQUIRE(grid.offset(2) == 4 + 6);
    REQUIRE(grid.offset(3) == grid.num_elements());

    // The index queries agree with the walk, and the bounds with the level
    // grids
    for (const auto& element : grid) {
        REQUIRE(grid.level_of(element.index()) == element.level());
        REQUIRE(grid.level_index(element.index())
                == grid.active_elements(element.level())
                       [element.index() - grid.offset(element.level())]);
        REQUIRE(grid.element_start(element.index()) == element.start());
        REQUIRE(grid.element_end(element.index()) == element.end());

        // The midpoint is held by the element, as is its start, on the
        // knot lines
        std::array<double, 2> middle{};

        for (std::size_t direction = 0; direction < 2; ++direction)
            middle[direction] =
                (element.start()[direction] + element.end()[direction]) / 2;

        REQUIRE(element_holding(grid, middle) == element.index());
        REQUIRE(element_holding(grid, element.start()) == element.index());

        // An active element covers itself on its level
        REQUIRE(element_covering(grid, element.level(),
                                 grid.level_index(element.index()))
                == element.index());
    }

    // Each cell of the finest level is covered by the active element that
    // holds its midpoint
    const int finest = grid.num_levels() - 1;
    const TensorGrid<double, 2>& cells = grid.level(finest);

    for (int cell = 0; cell < cells.num_elements(); ++cell) {
        const std::array<double, 2> start = cells.element_start(cell);
        const std::array<double, 2> end = cells.element_end(cell);
        const std::array<double, 2> middle{(start[0] + end[0]) / 2,
                                           (start[1] + end[1]) / 2};

        REQUIRE(element_covering(grid, finest, cell)
                == element_holding(grid, middle));
    }

    // The domain of a level holds its active elements and the parents of
    // the domain of the next, the refined elements 0 and 4 on level 0
    REQUIRE(grid.domain(0) == std::vector<int>{0, 1, 2, 3, 4, 5});
    REQUIRE(grid.domain(2) == grid.active_elements(2));

    for (int level = 0; level + 1 < grid.num_levels(); ++level) {
        std::vector<int> expected = grid.active_elements(level);
        const int columns =
            grid.level(level).knots(0).num_elements();
        const int fine_columns = 2 * columns;

        for (const int child : grid.domain(level + 1)) {
            const int row = child / fine_columns / 2;
            const int column = child % fine_columns / 2;
            expected.push_back(column + columns * row);
        }

        std::ranges::sort(expected);
        const auto repeats = std::ranges::unique(expected);
        expected.erase(repeats.begin(), repeats.end());

        REQUIRE(grid.domain(level) == expected);
    }

    // Grids with the same elements, on knots of other degrees, and a grid
    // refined elsewhere
    const HierarchicalGrid<double, 2> linear(TensorGrid<double, 2>(
        {KnotVector<double>(1, {0., 0., 1., 1., 3., 4., 4.}),
         KnotVector<double>(2, {0., 0., 0., .5, 2., 2., 2.})}));
    const HierarchicalGrid<double, 2> other =
        refine(refine(linear, std::vector<int>{0, 4}),
               std::vector<int>{6, 7});

    REQUIRE(grid == refine(once, std::vector<int>{7, 6}));
    REQUIRE(grid != once);
    REQUIRE(same_elements(grid, other));
    REQUIRE(!same_elements(grid, refine(once, std::vector<int>{6})));
}

TEST_CASE("Faces of a hierarchical grid tile its interior skeleton",
          "[grid]")
{
    const TensorGrid<double, 2> tensor(
        {KnotVector<double>(2, {0., 0., 0., 1., 1., 3., 4., 4., 4.}),
         KnotVector<double>(1, {0., 0., .5, 2., 2.})});
    const HierarchicalGrid<double, 2> coarse(tensor);

    // Without refinement, the faces are those of the tensor grid
    REQUIRE(coarse.num_faces() == tensor.num_faces());

    for (int face = 0; face < tensor.num_faces(); ++face) {
        REQUIRE(coarse.face_direction(face) == tensor.face_direction(face));
        REQUIRE(coarse.face_elements(face) == tensor.face_elements(face));
        REQUIRE(coarse.face_start(face) == tensor.face_start(face));
        REQUIRE(coarse.face_end(face) == tensor.face_end(face));
    }

    // Three levels, a fine element of level 2 next to a coarse one
    const HierarchicalGrid<double, 2> once =
        refine(coarse, std::vector<int>{0, 4});
    const HierarchicalGrid<double, 2> grid =
        refine(once, std::vector<int>{6, 7});
    std::vector<std::array<int, 2>> pairs;
    double length = 0.;

    for (int face = 0; face < grid.num_faces(); ++face) {
        const std::size_t direction = grid.face_direction(face);
        const std::size_t across = 1 - direction;
        const auto [before, after] = grid.face_elements(face);
        const std::array<double, 2> start = grid.face_start(face);
        const std::array<double, 2> end = grid.face_end(face);

        // The face lies on the end side of the element before it and the
        // start side of the element after it, within both
        REQUIRE(start[direction] == end[direction]);
        REQUIRE(start[direction] == grid.element_end(before)[direction]);
        REQUIRE(start[direction] == grid.element_start(after)[direction]);
        REQUIRE(start[across] < end[across]);

        for (const int element : {before, after}) {
            REQUIRE(start[across] >= grid.element_start(element)[across]);
            REQUIRE(end[across] <= grid.element_end(element)[across]);
        }

        pairs.push_back({before, after});
        length += end[across] - start[across];
    }

    // Each pair of neighbors shares one face, and the sides of the elements
    // count every face twice and the edges of the domain once
    std::ranges::sort(pairs);
    REQUIRE(std::ranges::adjacent_find(pairs) == pairs.end());

    double perimeters = 0.;

    for (int element = 0; element < grid.num_elements(); ++element) {
        const std::array<double, 2> start = grid.element_start(element);
        const std::array<double, 2> end = grid.element_end(element);
        perimeters += 2. * (end[0] - start[0] + end[1] - start[1]);
    }

    REQUIRE_THAT(perimeters, WithinAbs(2. * length + 2. * (4. + 2.), 1e-12));
}

TEST_CASE("Refinement rejects what it cannot refine", "[grid]")
{
    const KnotVector<double> knots(1, {0., 0., 1., 1.});
    HierarchicalGrid<double, 3> grid(
        TensorGrid<double, 3>({knots, knots, knots}));

    REQUIRE_THROWS_AS(refine(grid, std::vector<int>{-1}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(refine(grid, std::vector<int>{1}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(refine_admissible(grid, std::vector<int>{1}, true),
                      std::invalid_argument);

    // Refining the last element ten times leaves 8^10 = 2^30 elements on
    // the finest level, so the next one would have 2^33, beyond int
    for (int step = 0; step < 10; ++step)
        grid = refine(grid, std::vector<int>{grid.num_elements() - 1});

    REQUIRE_THROWS_AS(
        refine(grid, std::vector<int>{grid.num_elements() - 1}),
        std::invalid_argument);
}
