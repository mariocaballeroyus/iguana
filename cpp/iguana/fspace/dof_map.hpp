/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_FSPACE_DOF_MAP_HPP
#define IGUANA_FSPACE_DOF_MAP_HPP

#include <cstddef>
#include <span>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Global degrees of freedom of the functions active on each element
 *
 * The degrees of freedom are stored element after element, and an offset
 * marks where those of each element start. Elements may list different
 * numbers of them, and an element listing none, such as a cell outside the
 * physical domain, contributes no unknowns
 */
class DofMap
{
public:
    /**
     * @brief Constructs the map from its flat arrays
     *
     * @param num_dofs Number of degrees of freedom of the space
     * @param offsets First degree of freedom of each element, followed by
     *        the number listed, with size num_elements + 1
     * @param dofs Degrees of freedom, element after element
     *
     * @throws std::invalid_argument If the offsets do not run from zero to
     *         the number of listed degrees of freedom without decreasing, or
     *         if a degree of freedom lies outside [0, num_dofs)
     */
    DofMap(int num_dofs, Eigen::VectorXi offsets, Eigen::VectorXi dofs);

    /// @brief Number of degrees of freedom of the space
    constexpr int num_dofs() const noexcept
    { return num_dofs_; }

    /// @brief Number of elements, in the numbering of the domain
    constexpr int num_elements() const noexcept
    { return static_cast<int>(offsets_.size()) - 1; }

    /**
     * @brief Degrees of freedom of the functions active on an element
     *
     * @param element Element index, in the numbering of the domain
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr std::span<const int> dofs_on_element(int element) const noexcept
    {
        const int first = offsets_(element);
        const int count = offsets_(element + 1) - first;
        return {dofs_.data() + first, static_cast<std::size_t>(count)};
    }

private:
    /// @brief Number of degrees of freedom of the space
    int num_dofs_;

    /// @brief First degree of freedom of each element, followed by the
    ///        number listed
    Eigen::VectorXi offsets_;

    /// @brief Degrees of freedom, element after element
    Eigen::VectorXi dofs_;
};

} // namespace iguana

#endif // IGUANA_FSPACE_DOF_MAP_HPP
