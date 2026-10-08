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

namespace
{

using iguana::CellClassification;
using iguana::CellType;
using iguana::GhostFaces;
using iguana::KnotVector;
using iguana::TensorGrid;

} // namespace

TEST_CASE("The ghost faces surround the cut and outside cells",
          "[ghost_faces]")
{
    // Three by three elements over uneven spans, numbered from the lower
    // left, with the first direction running fastest:
    //
    //   O O O
    //   I C O
    //   I I O
    const TensorGrid<double, 2> grid(
        {KnotVector<double>(1, {0., 0., .2, .6, 1., 1.}),
         KnotVector<double>(1, {0., 0., .3, .5, 1., 1.})});

    const CellClassification<double, 2> classification(
        {CellType::inside, CellType::inside, CellType::outside,
         CellType::inside, CellType::cut, CellType::outside,
         CellType::outside, CellType::outside, CellType::outside});

    const GhostFaces<double, 2> ghost(grid, classification);

    // The faces of the cut cell and those between outside cells, but none
    // between inside cells, nor between an inside and an outside cell, as
    // direction, cell before and cell after, ordered by the cell before
    const std::vector<std::array<int, 3>> expected{
        {1, 1, 4}, {1, 2, 5}, {0, 3, 4}, {0, 4, 5},
        {1, 4, 7}, {1, 5, 8}, {0, 6, 7}, {0, 7, 8}};
    std::vector<std::array<int, 3>> found;

    for (const GhostFaces<double, 2>::Face& face : ghost.faces())
        found.push_back({face.direction, face.before, face.after});

    REQUIRE(found == expected);

    // Each face lies on the knot line at the end of the cell before it,
    // across the span of that cell
    const std::array<std::vector<double>, 2> lines = grid.lines();

    for (const GhostFaces<double, 2>::Face& face : ghost.faces()) {
        const std::array<int, 2> cell{face.before % 3, face.before / 3};

        for (std::size_t axis = 0; axis < 2; ++axis) {
            const int offset = static_cast<int>(axis) == face.direction;

            REQUIRE(face.start[axis] == lines[axis][cell[axis] + offset]);
            REQUIRE(face.end[axis] == lines[axis][cell[axis] + 1]);
        }
    }

    // One cell type for nine elements
    const CellClassification<double, 2> one_cell({CellType::inside});

    REQUIRE_THROWS_AS((GhostFaces<double, 2>(grid, one_cell)),
                      std::invalid_argument);
}

TEST_CASE("Every face between the outside cells of a volume is a ghost face",
          "[ghost_faces]")
{
    const KnotVector<double> halves(1, {0., 0., .5, 1., 1.});
    const TensorGrid<double, 3> grid({halves, halves, halves});

    const GhostFaces<double, 3> ghost(
        grid, CellClassification<double, 3>(
                  std::vector<CellType>(8, CellType::outside)));

    // Four faces normal to each direction
    REQUIRE(ghost.num_faces() == 12);
}
