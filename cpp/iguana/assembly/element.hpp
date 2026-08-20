/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_ASSEMBLY_ELEMENT_HPP
#define IGUANA_ASSEMBLY_ELEMENT_HPP

#include<concepts>

#include "iguana/assembly/element_data.hpp"

namespace iguana
{

/**
 * @brief A domain integrand of a problem, one term of its weak form.
 *
 * An Element fills the local residual and jacobian of one domain
 * element from the prepared data. The assembler chooses what it assembles, 
 * calling the residual alone or the jacobian beside it.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the system.
 */
template<int d, std::floating_point T>
class Element
{
public:
    virtual ~Element() = default;

    /// @brief The quantities the weak form reads, combined by the
    ///        assembler over the integrands of the problem.
    virtual UpdateFlags updates() const noexcept = 0;

    /**
     * @brief Adds the local residual of one domain element.
     *
     * @param data The prepared data of the element.
     * @param residual Local residual, of size (num_active,), zeroed by
     *        the assembler and added to by every integrand.
     */
    virtual void residual(const ElementData<d, T>& data,
                          Eigen::VectorX<T>& residual) const = 0;

    /**
     * @brief Adds the local jacobian of one domain element.
     *
     * @param data The prepared data of the element.
     * @param jacobian Local jacobian, of size (num_active,num_active),
     *        zeroed by the assembler and added to by every integrand.
     */
    virtual void jacobian(const ElementData<d, T>& data,
                          Eigen::MatrixX<T>& jacobian) const = 0;
};

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ELEMENT_HPP
