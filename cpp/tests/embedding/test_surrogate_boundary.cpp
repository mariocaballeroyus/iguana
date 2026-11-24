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
using iguana::CellType;
using iguana::CellClassification;
using iguana::KnotVector;
using iguana::SurrogateBoundary;
using iguana::TensorGrid;

/// @brief Measure of each face and its first moment, the integral of x_k
///        n_k over the faces flat in each direction k, which the divergence
///        theorem equates with the volume they bound
template<std::size_t d>
struct Moments
{
    double measure = 0.;
    std::array<double, d> flux{};
};

template<std::size_t d>
Moments<d> moments(const SurrogateBoundary<double, d>& boundary)
{
    Moments<d> result;

    for (int element = 0; element < boundary.num_elements(); ++element) {
        for (const typename SurrogateBoundary<double, d>::Face& face :
             boundary.faces_on_element(element)) {
            const std::size_t direction =
                static_cast<std::size_t>(face.direction);
            double measure = 1.;

            for (std::size_t other = 0; other < d; ++other)
                if (other != direction)
                    measure *= face.end[other] - face.start[other];

            // x_k is constant on the face and n_k is its side
            result.measure += measure;
            result.flux[direction] +=
                face.side * face.start[direction] * measure;
        }
    }

    return result;
}

} // namespace

TEST_CASE("The surrogate boundary bounds the inside cells",
          "[surrogate_boundary]")
{
    // Four by three elements over uneven spans, the last column cut, and
    // the cell at (1, 1) outside and enclosed by inside cells
    const TensorGrid<double, 2> grid(
        {KnotVector<double>(1, {0., 0., .1, .4, .7, 1., 1.}),
         KnotVector<double>(1, {0., 0., .3, .5, 1., 1.})});

    std::vector<CellType> cell_types(12, CellType::inside);

    for (const int element : {3, 7, 11})
        cell_types[element] = CellType::cut;

    cell_types[5] = CellType::outside;

    const CellClassification<double, 2> classification(cell_types);
    const SurrogateBoundary<double, 2> boundary(grid, classification);

    // Only inside cells hold faces
    for (int element = 0; element < boundary.num_elements(); ++element)
        if (!boundary.faces_on_element(element).empty())
            REQUIRE(classification.cell_type(element) == CellType::inside);

    // The surrogate domain is [0, .7] x [0, 1] without the hole
    // [.1, .4] x [.3, .5], and its boundary runs along the edge of the grid
    // too
    const Moments<2> result = moments(boundary);

    REQUIRE_THAT(result.measure, WithinAbs(2. * (.7 + 1.) + 2. * (.3 + .2),
                                           1e-14));
    REQUIRE_THAT(result.flux[0], WithinAbs(.7 - .06, 1e-14));
    REQUIRE_THAT(result.flux[1], WithinAbs(.7 - .06, 1e-14));

    // One cell type for twelve elements
    const CellClassification<double, 2> one_cell({CellType::inside});

    REQUIRE_THROWS_AS((SurrogateBoundary<double, 2>(grid, one_cell)),
                      std::invalid_argument);
}

TEST_CASE("The surrogate boundary of a hierarchical grid bounds the same "
          "region in stretches", "[surrogate_boundary]")
{
    // The grid and cells of the planar test: the last column cut and the
    // cell at (1, 1) outside, enclosed by inside cells
    const TensorGrid<double, 2> tensor(
        {KnotVector<double>(1, {0., 0., .1, .4, .7, 1., 1.}),
         KnotVector<double>(1, {0., 0., .3, .5, 1., 1.})});
    std::vector<CellType> cell_types(12, CellType::inside);

    for (const int element : {3, 7, 11})
        cell_types[element] = CellType::cut;

    cell_types[5] = CellType::outside;

    const CellClassification<double, 2> classification(cell_types);
    const iguana::HierarchicalGrid<double, 2> coarse(tensor);
    const SurrogateBoundary<double, 2> expected(tensor, classification);
    const SurrogateBoundary<double, 2> unrefined(coarse, classification);

    // Without refinement, each cell holds the faces of the tensor grid
    for (int element = 0; element < 12; ++element) {
        const auto faces = unrefined.faces_on_element(element);
        const auto wanted = expected.faces_on_element(element);

        REQUIRE(faces.size() == wanted.size());

        for (std::size_t entry = 0; entry < faces.size(); ++entry) {
            REQUIRE(faces[entry].direction == wanted[entry].direction);
            REQUIRE(faces[entry].side == wanted[entry].side);
            REQUIRE(faces[entry].start == wanted[entry].start);
            REQUIRE(faces[entry].end == wanted[entry].end);
        }
    }

    // The hole refined, its children outside like it, and an inside and a
    // cut neighbor too: the boundary runs in more stretches around the same
    // region
    const iguana::HierarchicalGrid<double, 2> around =
        iguana::refine(coarse, std::vector<int>{5, 6, 7});
    std::vector<CellType> inherited;

    // Each element takes the type of the coarse cell it lies in
    for (int element = 0; element < around.num_elements(); ++element) {
        const int shift = around.level_of(element);
        const int index = around.level_index(element);
        const int columns = 4 << shift;
        const int column = (index % columns) >> shift;
        const int row = (index / columns) >> shift;

        inherited.push_back(
            cell_types[static_cast<std::size_t>(column + 4 * row)]);
    }

    const Moments<2> same = moments(SurrogateBoundary<double, 2>(
        around, CellClassification<double, 2>(inherited)));
    const Moments<2> region = moments(expected);

    REQUIRE_THAT(same.measure, WithinAbs(region.measure, 1e-14));
    REQUIRE_THAT(same.flux[0], WithinAbs(region.flux[0], 1e-14));
    REQUIRE_THAT(same.flux[1], WithinAbs(region.flux[1], 1e-14));

    // The hole refined with its lower children inside: it shrinks to
    // [.1, .4] x [.4, .5], and the cell left of it holds only the stretch
    // facing the outside child
    const iguana::HierarchicalGrid<double, 2> half =
        iguana::refine(coarse, std::vector<int>{5});
    std::vector<CellType> halved(cell_types);
    halved.erase(halved.begin() + 5);
    halved.insert(halved.end(), {CellType::inside, CellType::inside,
                                 CellType::outside, CellType::outside});

    const SurrogateBoundary<double, 2> shrunk(
        half, CellClassification<double, 2>(halved));
    const Moments<2> smaller = moments(shrunk);

    REQUIRE_THAT(smaller.measure,
                 WithinAbs(2. * (.7 + 1.) + 2. * (.3 + .1), 1e-14));
    REQUIRE_THAT(smaller.flux[0], WithinAbs(.7 - .03, 1e-14));
    REQUIRE_THAT(smaller.flux[1], WithinAbs(.7 - .03, 1e-14));

    const auto left = shrunk.faces_on_element(4);

    REQUIRE(left.size() == 2);
    REQUIRE(left[1].direction == 0);
    REQUIRE(left[1].side == 1);
    REQUIRE_THAT(left[1].start[1], WithinAbs(.4, 1e-15));
    REQUIRE_THAT(left[1].end[1], WithinAbs(.5, 1e-15));
}

TEST_CASE("The surrogate boundary of one inside cell in a volume",
          "[surrogate_boundary]")
{
    const KnotVector<double> halves(1, {0., 0., .5, 1., 1.});
    const TensorGrid<double, 3> grid({halves, halves, halves});

    std::vector<CellType> cell_types(8, CellType::outside);
    cell_types[0] = CellType::inside;

    const SurrogateBoundary<double, 3> boundary(
        grid, CellClassification<double, 3>(cell_types));

    // The six faces of the cube [0, .5]^3
    const Moments<3> result = moments(boundary);

    REQUIRE(boundary.num_faces() == 6);
    REQUIRE_THAT(result.measure, WithinAbs(6. * .25, 1e-14));

    for (const double flux : result.flux)
        REQUIRE_THAT(flux, WithinAbs(.125, 1e-14));
}
