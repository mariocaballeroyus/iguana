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
using iguana::HierarchicalDomain;
using iguana::KnotVector;
using iguana::TensorDomain;

/// @brief Parameters at which an element starts, then those at which it
///        ends
using Box = std::array<std::array<double, 2>, 2>;

} // namespace

TEST_CASE("Refined elements tile the domain", "[domain]")
{
    // Six elements over [0, 4] x [0, 2], with a repeated interior knot and
    // uneven spans
    const HierarchicalDomain<double, 2> coarse(TensorDomain<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., 1., 1., 3., 4., 4., 4.}),
         KnotVector<double>(1, {0., 0., .5, 2., 2.})}));

    // Each distinct mark trades an element for its four children, the
    // first ones on a new level
    const HierarchicalDomain<double, 2> once =
        refine(coarse, std::vector<int>{0, 4, 4});

    REQUIRE(once.num_levels() == 2);
    REQUIRE(once.num_elements() == 6 + 2 * 3);

    // Every coarse element and a child, which leaves level 0 empty
    const HierarchicalDomain<double, 2> domain =
        refine(once, std::vector<int>{0, 1, 2, 3, 4});

    REQUIRE(domain.num_levels() == 3);
    REQUIRE(domain.active_elements(0).empty());
    REQUIRE(domain.num_elements() == 12 + 5 * 3);

    std::vector<Box> boxes;
    int level = 0;
    int position = 0;

    for (const auto& element : domain) {
        REQUIRE(element.index() == static_cast<int>(boxes.size()));
        REQUIRE(element.level() >= level);

        if (element.level() > level) {
            level = element.level();
            position = 0;
        }

        // The element of the level domain it stands for, whose walk is the
        // reference
        const int active = domain.active_elements(level)[position];
        ++position;

        for (const auto& reference : domain.level(level)) {
            if (reference.index() != active)
                continue;

            REQUIRE(element.first_active() == reference.first_active());
            REQUIRE(element.start() == reference.start());
            REQUIRE(element.end() == reference.end());
        }

        boxes.push_back({element.start(), element.end()});
    }

    REQUIRE(static_cast<int>(boxes.size()) == domain.num_elements());

    // Pairwise disjoint boxes whose areas add up to that of the domain
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

TEST_CASE("Refinement rejects what it cannot refine", "[domain]")
{
    const KnotVector<double> knots(1, {0., 0., 1., 1.});
    HierarchicalDomain<double, 3> domain(
        TensorDomain<double, 3>({knots, knots, knots}));

    REQUIRE_THROWS_AS(refine(domain, std::vector<int>{-1}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(refine(domain, std::vector<int>{1}),
                      std::invalid_argument);

    // Refining the last element ten times leaves 8^10 = 2^30 elements on
    // the finest level, so the next one would have 2^33, beyond int
    for (int step = 0; step < 10; ++step)
        domain = refine(domain, std::vector<int>{domain.num_elements() - 1});

    REQUIRE_THROWS_AS(
        refine(domain, std::vector<int>{domain.num_elements() - 1}),
        std::invalid_argument);
}
