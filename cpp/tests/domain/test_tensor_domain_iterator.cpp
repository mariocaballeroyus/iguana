/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::KnotVector;
using iguana::TensorDomain;

/// @brief Domain mixing degrees, a repeated interior knot and an unclamped
///        direction
TensorDomain<double, 3> mixed()
{
    return TensorDomain<double, 3>(
        {KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 3., 3.}),
         KnotVector<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
         KnotVector<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5})});
}

/// @brief Element of each direction, split from the flat index: the
///        reference the walk is checked against
template<std::size_t d>
std::array<int, d> element_of(const TensorDomain<double, d>& domain,
                              int element)
{
    std::array<int, d> indices{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const int count = domain.knots(direction).num_elements();
        indices[direction] = element % count;
        element /= count;
    }

    return indices;
}

} // namespace

TEST_CASE("The walk reaches every element once, in order", "[domain]")
{
    const TensorDomain<double, 3> domain = mixed();

    int visited = 0;
    double volume = 0.;

    for (const auto& element : domain) {
        REQUIRE(element.index() == visited);
        ++visited;

        // The walk advances the element of each direction, so it must
        // agree with splitting the flat index anew
        const std::array<int, 3> indices =
            element_of(domain, element.index());

        double cell = 1.;

        for (std::size_t direction = 0; direction < 3; ++direction) {
            const KnotVector<double>& knots = domain.knots(direction);

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

    REQUIRE(visited == domain.num_elements());

    // The element boxes fill the parameter box
    double box = 1.;

    for (std::size_t direction = 0; direction < 3; ++direction)
        box *= domain.knots(direction).domain_end()
               - domain.knots(direction).domain_start();

    REQUIRE_THAT(volume, WithinAbs(box, 1e-12));

    // A second walk starts over
    int again = 0;

    for (const auto& element : domain)
        REQUIRE(element.index() == again++);

    REQUIRE(again == domain.num_elements());
}
