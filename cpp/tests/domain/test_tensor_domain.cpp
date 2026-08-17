/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "iguana/domain/tensor_domain.hpp"

#include<array>
#include<cmath>

#include<catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;
using iguana::TensorProductBSpline;

/// @brief A 3D basis mixing degrees, a repeated interior knot and a
///        direction that is not open.
TensorProductBSpline<3, double> mixed()
{
    return TensorProductBSpline<3, double>(
        {BSpline<double>(2, {0., 0., 0., 1., 2., 3., 3., 3.}),
         BSpline<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
         BSpline<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5})});
}

/// @brief The element of every direction, split from the flat index:
///        the independent reference the iterator is pinned against.
template<int d>
std::array<int, d> element_of(const TensorProductBSpline<d, double>& basis,
                              int element)
{
    std::array<int, d> each{};

    for (int k = 0; k < d; ++k) {
        const int count = basis.axis(k).num_elements();

        each[k] = element % count;
        element /= count;
    }

    return each;
}

} // namespace

TEST_CASE("The walk reaches every element once, in order", "[domain]")
{
    const iguana::TensorDomain<3, double> domain(mixed());
    const TensorProductBSpline<3, double>& basis = domain.basis();

    int visited = 0;
    double volume = 0.;

    for (const auto& element : domain) {
        REQUIRE(element.index() == visited);
        ++visited;

        // The walk advances the element of every direction, so it has
        // to agree with splitting the flat index anew
        const std::array<int, 3> each = element_of(basis, element.index());

        double cell = 1.;

        for (int k = 0; k < 3; ++k) {
            const BSpline<double>& axis = basis.axis(k);

            REQUIRE(element.first_active()[k]
                    == axis.first_active(each[k]));
            REQUIRE(element.start()[k] == axis.element_start(each[k]));
            REQUIRE(element.end()[k] == axis.element_end(each[k]));

            cell *= element.end()[k] - element.start()[k];
        }

        volume += cell;
    }

    REQUIRE(visited == domain.num_elements());

    // The element boxes fill the whole domain
    double whole = 1.;

    for (int k = 0; k < 3; ++k) {
        const BSpline<double>& axis = basis.axis(k);

        whole *= axis.element_end(axis.num_elements() - 1)
                 - axis.element_start(0);
    }

    REQUIRE(std::abs(volume - whole) < 1e-12);

    // A second walk starts over
    int again = 0;

    for (const auto& element : domain) {
        REQUIRE(element.index() == again);
        ++again;
    }

    REQUIRE(again == domain.num_elements());
}

TEST_CASE("A single direction degenerates to the univariate elements",
          "[domain]")
{
    const BSpline<double> axis(2, {0., 0., 0., 1., 2., 3., 3., 3.});
    const iguana::TensorDomain<1, double> domain(
        TensorProductBSpline<1, double>({axis}));

    int visited = 0;

    for (const auto& element : domain) {
        REQUIRE(element.index() == visited);
        REQUIRE(element.first_active()[0] == axis.first_active(visited));
        REQUIRE(element.start()[0] == axis.element_start(visited));
        REQUIRE(element.end()[0] == axis.element_end(visited));

        ++visited;
    }

    REQUIRE(visited == axis.num_elements());
}
