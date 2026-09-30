/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "dof_map.hpp"

#include <stdexcept>
#include <utility>

namespace iguana
{

DofMap::DofMap(int num_dofs, Eigen::VectorXi offsets, Eigen::VectorXi dofs)
    : num_dofs_(num_dofs),
      offsets_(std::move(offsets)),
      dofs_(std::move(dofs))
{
    // Element k lists the degrees of freedom from offsets(k) up to
    // offsets(k + 1), none when the two are equal
    const Eigen::Index count = offsets_.size() - 1;
    const bool ordered =
        count >= 0 && offsets_(0) == 0 && offsets_(count) == dofs_.size() &&
        (offsets_.tail(count).array() >= offsets_.head(count).array()).all();

    if (!ordered)
        throw std::invalid_argument("DofMap: "
                                    "the offsets must run from zero to the "
                                    "number of listed degrees of freedom "
                                    "without decreasing");

    if ((dofs_.array() < 0).any() || (dofs_.array() >= num_dofs_).any())
        throw std::invalid_argument("DofMap: "
                                    "the degrees of freedom must lie in "
                                    "[0, num_dofs)");
}

} // namespace iguana
