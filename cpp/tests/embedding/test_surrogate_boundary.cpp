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
using iguana::EmbeddedDomain;
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

    const EmbeddedDomain<double, 2> domain(cell_types);
    const SurrogateBoundary<double, 2> boundary(grid, domain);

    // Only inside cells hold faces
    for (int element = 0; element < boundary.num_elements(); ++element)
        if (!boundary.faces_on_element(element).empty())
            REQUIRE(domain.cell_type(element) == CellType::inside);

    // The surrogate domain is [0, .7] x [0, 1] without the hole
    // [.1, .4] x [.3, .5], and its boundary runs along the edge of the grid
    // too
    const Moments<2> result = moments(boundary);

    REQUIRE_THAT(result.measure, WithinAbs(2. * (.7 + 1.) + 2. * (.3 + .2),
                                           1e-14));
    REQUIRE_THAT(result.flux[0], WithinAbs(.7 - .06, 1e-14));
    REQUIRE_THAT(result.flux[1], WithinAbs(.7 - .06, 1e-14));

    // One cell type for twelve elements
    const EmbeddedDomain<double, 2> one_cell({CellType::inside});

    REQUIRE_THROWS_AS((SurrogateBoundary<double, 2>(grid, one_cell)),
                      std::invalid_argument);
}

TEST_CASE("The surrogate boundary of one inside cell in a volume",
          "[surrogate_boundary]")
{
    const KnotVector<double> halves(1, {0., 0., .5, 1., 1.});
    const TensorGrid<double, 3> grid({halves, halves, halves});

    std::vector<CellType> cell_types(8, CellType::outside);
    cell_types[0] = CellType::inside;

    const SurrogateBoundary<double, 3> boundary(
        grid, EmbeddedDomain<double, 3>(cell_types));

    // The six faces of the cube [0, .5]^3
    const Moments<3> result = moments(boundary);

    REQUIRE(boundary.num_faces() == 6);
    REQUIRE_THAT(result.measure, WithinAbs(6. * .25, 1e-14));

    for (const double flux : result.flux)
        REQUIRE_THAT(flux, WithinAbs(.125, 1e-14));
}
