/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cstddef>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::TensorBSpline;
using iguana::TensorDomain;

/// @brief Basis mixing degrees, a repeated interior knot and an unclamped
///        direction
TensorBSpline<double, 3> mixed()
{
    return TensorBSpline<double, 3>(
        {BSpline<double>(2, {0., 0., 0., 1., 2., 3., 3., 3.}),
         BSpline<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
         BSpline<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5})});
}

/// @brief Domain over a basis, whose control points do not matter here
template<std::size_t d>
TensorDomain<double, d> domain_of(TensorBSpline<double, d> basis)
{
    const int num_functions = basis.num_functions();

    return TensorDomain<double, d>(iguana::Patch<double, d>(
        std::move(basis),
        iguana::PointMatrix<double>::Zero(num_functions, 3)));
}

/// @brief Element of each direction, split from the flat index: the
///        reference the walk is checked against
template<std::size_t d>
std::array<int, d> element_of(const TensorBSpline<double, d>& basis,
                              int element)
{
    std::array<int, d> indices{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const int count = basis.axis(direction).num_elements();
        indices[direction] = element % count;
        element /= count;
    }

    return indices;
}

} // namespace

TEST_CASE("The walk reaches every element once, in order", "[domain]")
{
    const TensorDomain<double, 3> domain = domain_of(mixed());
    const TensorBSpline<double, 3>& basis = domain.basis();

    int visited = 0;
    double volume = 0.;

    for (const auto& element : domain) {
        REQUIRE(element.index() == visited);
        ++visited;

        // The walk advances the element of each direction, so it must
        // agree with splitting the flat index anew
        const std::array<int, 3> indices =
            element_of(basis, element.index());

        double cell = 1.;

        for (std::size_t direction = 0; direction < 3; ++direction) {
            const BSpline<double>& axis = basis.axis(direction);

            REQUIRE(element.first_active()[direction]
                    == axis.first_active(indices[direction]));
            REQUIRE(element.start()[direction]
                    == axis.element_start(indices[direction]));
            REQUIRE(element.end()[direction]
                    == axis.element_end(indices[direction]));

            cell *= element.end()[direction] - element.start()[direction];
        }

        volume += cell;
    }

    REQUIRE(visited == domain.num_elements());

    // The element boxes fill the parameter box
    double box = 1.;

    for (std::size_t direction = 0; direction < 3; ++direction) {
        const BSpline<double>& axis = basis.axis(direction);

        box *= axis.element_end(axis.num_elements() - 1)
               - axis.element_start(0);
    }

    REQUIRE_THAT(volume, WithinAbs(box, 1e-12));

    // A second walk starts over
    int again = 0;

    for (const auto& element : domain)
        REQUIRE(element.index() == again++);

    REQUIRE(again == domain.num_elements());
}
