/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::CellClassification;
using iguana::CellType;
using iguana::JumpFaces;
using iguana::KnotVector;
using iguana::TensorGrid;

} // namespace

TEST_CASE("The jump faces separate cells whose fractions may differ",
          "[jump_faces]")
{
    // Three by three elements, numbered from the lower left, with the first
    // direction running fastest:
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

    const JumpFaces<double, 2> jump(grid, classification);

    // The faces between cells of different types, but none between two
    // inside or two outside cells, as direction, cell before and cell
    // after, ordered by the cell before
    const std::vector<std::array<int, 3>> expected{
        {0, 1, 2}, {1, 1, 4}, {0, 3, 4}, {1, 3, 6}, {0, 4, 5}, {1, 4, 7}};
    std::vector<std::array<int, 3>> found;

    for (const JumpFaces<double, 2>::Face& face : jump.faces())
        found.push_back({face.direction, face.before, face.after});

    REQUIRE(found == expected);

    // One cell type for nine elements
    const CellClassification<double, 2> one_cell({CellType::inside});

    REQUIRE_THROWS_AS((JumpFaces<double, 2>(grid, one_cell)),
                      std::invalid_argument);
}

TEST_CASE("With cells inside or outside, the jump faces are the surrogate "
          "boundary", "[jump_faces]")
{
    // A block of two by two inside cells in the middle of four by four, so
    // that the surrogate boundary stays off the edge of the grid
    const KnotVector<double> quarters(1, {0., 0., .25, .5, .75, 1., 1.});
    const TensorGrid<double, 2> grid({quarters, quarters});

    std::vector<CellType> cell_types(16, CellType::outside);

    for (const int element : {5, 6, 9, 10})
        cell_types[element] = CellType::inside;

    const CellClassification<double, 2> classification(cell_types);
    const JumpFaces<double, 2> jump(grid, classification);
    const iguana::SurrogateBoundary<double, 2> surrogate(grid,
                                                         classification);

    REQUIRE(jump.num_faces() == 8);
    REQUIRE(surrogate.num_faces() == 8);

    // Two cut cells may hold different fractions, and are both kept
    std::vector<CellType> two_cut(4, CellType::cut);
    const KnotVector<double> halves(1, {0., 0., .5, 1., 1.});

    REQUIRE(JumpFaces<double, 2>(TensorGrid<double, 2>({halves, halves}),
                                 CellClassification<double, 2>(two_cut))
                .num_faces() == 4);
}
