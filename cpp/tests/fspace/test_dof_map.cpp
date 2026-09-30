/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::DofMap;

/// @brief Integer vector from its entries
Eigen::VectorXi integers(const std::vector<int>& entries)
{
    return Eigen::Map<const Eigen::VectorXi>(entries.data(), entries.size());
}

} // namespace

TEST_CASE("The flat arrays must list valid degrees of freedom per element",
          "[fspace]")
{
    const Eigen::VectorXi offsets = integers({0, 2, 2, 3});
    const Eigen::VectorXi dofs = integers({0, 1, 2});

    // An element may list no degrees of freedom
    REQUIRE_NOTHROW(DofMap(3, offsets, dofs));

    REQUIRE_THROWS_AS(DofMap(3, integers({0, 2, 1, 3}), dofs),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(DofMap(2, offsets, dofs), std::invalid_argument);
}
