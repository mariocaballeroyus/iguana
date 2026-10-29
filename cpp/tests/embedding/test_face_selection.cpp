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
using iguana::KnotVector;
using iguana::TensorGrid;

} // namespace

TEST_CASE("The ghost faces surround the cut and outside cells",
          "[face_selection]")
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

    // Each face as direction, cell before and cell after, in the numbering
    // of the grid
    const auto described = [&grid](const std::vector<int>& faces)
    {
        std::vector<std::array<int, 3>> found;

        for (const int face : faces) {
            const std::array<int, 2> cells = grid.face_elements(face);

            found.push_back({static_cast<int>(grid.face_direction(face)),
                             cells[0], cells[1]});
        }

        return found;
    };

    // The faces of the cut cell and those between outside cells, but none
    // between inside cells, nor between an inside and an outside cell
    REQUIRE(described(iguana::ghost_faces(grid, classification))
            == std::vector<std::array<int, 3>>{
                {0, 3, 4}, {0, 4, 5}, {0, 6, 7}, {0, 7, 8},
                {1, 1, 4}, {1, 2, 5}, {1, 4, 7}, {1, 5, 8}});

    // Either kind alone, the two sharing them
    REQUIRE(described(iguana::ghost_faces(grid, classification,
                                          iguana::GhostFaces::cut))
            == std::vector<std::array<int, 3>>{
                {0, 3, 4}, {0, 4, 5}, {1, 1, 4}, {1, 4, 7}});
    REQUIRE(described(iguana::ghost_faces(grid, classification,
                                          iguana::GhostFaces::outside))
            == std::vector<std::array<int, 3>>{
                {0, 6, 7}, {0, 7, 8}, {1, 2, 5}, {1, 5, 8}});

    // One cell type for nine elements
    const CellClassification<double, 2> one_cell({CellType::inside});

    REQUIRE_THROWS_AS(iguana::ghost_faces(grid, one_cell),
                      std::invalid_argument);
}

TEST_CASE("Every face between the outside cells of a volume is a ghost face",
          "[face_selection]")
{
    const KnotVector<double> halves(1, {0., 0., .5, 1., 1.});
    const TensorGrid<double, 3> grid({halves, halves, halves});

    const std::vector<int> ghost = iguana::ghost_faces(
        grid, CellClassification<double, 3>(
                  std::vector<CellType>(8, CellType::outside)));

    // Four faces normal to each direction
    REQUIRE(ghost.size() == 12);
}

TEST_CASE("The jump faces separate cells whose fractions may differ",
          "[face_selection]")
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

    const std::vector<int> jump = iguana::jump_faces(grid, classification);

    // The faces between cells of different types, but none between two
    // inside or two outside cells, as direction, cell before and cell
    // after, in the numbering of the grid
    const std::vector<std::array<int, 3>> expected{
        {0, 1, 2}, {0, 3, 4}, {0, 4, 5}, {1, 1, 4}, {1, 3, 6}, {1, 4, 7}};
    std::vector<std::array<int, 3>> found;

    for (const int face : jump) {
        const std::array<int, 2> cells = grid.face_elements(face);

        found.push_back({static_cast<int>(grid.face_direction(face)),
                         cells[0], cells[1]});
    }

    REQUIRE(found == expected);

    // One cell type for nine elements
    const CellClassification<double, 2> one_cell({CellType::inside});

    REQUIRE_THROWS_AS(iguana::jump_faces(grid, one_cell),
                      std::invalid_argument);
}

TEST_CASE("With cells inside or outside, the jump faces are the surrogate "
          "boundary", "[face_selection]")
{
    // A block of two by two inside cells in the middle of four by four, so
    // that the surrogate boundary stays off the edge of the grid
    const KnotVector<double> quarters(1, {0., 0., .25, .5, .75, 1., 1.});
    const TensorGrid<double, 2> grid({quarters, quarters});

    std::vector<CellType> cell_types(16, CellType::outside);

    for (const int element : {5, 6, 9, 10})
        cell_types[element] = CellType::inside;

    const CellClassification<double, 2> classification(cell_types);
    const std::vector<int> jump = iguana::jump_faces(grid, classification);
    const iguana::SurrogateBoundary<double, 2> surrogate(grid,
                                                         classification);

    REQUIRE(jump.size() == 8);
    REQUIRE(surrogate.num_faces() == 8);

    // Two cut cells may hold different fractions, and are both kept
    std::vector<CellType> two_cut(4, CellType::cut);
    const KnotVector<double> halves(1, {0., 0., .5, 1., 1.});

    REQUIRE(iguana::jump_faces(TensorGrid<double, 2>({halves, halves}),
                               CellClassification<double, 2>(two_cut))
                .size() == 4);
}
