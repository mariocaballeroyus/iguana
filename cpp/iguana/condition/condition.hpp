/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_CONDITION_HPP
#define IGUANA_CONDITION_CONDITION_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"

namespace iguana
{

/**
 * @brief Condition on a boundary, giving its stiffness and load on the
 *        part of the boundary in one element, from the values at the points
 *        there
 *
 * A condition reads from ElementValues the values its flags() request and
 * integrates along the boundary with boundary weights, the quadrature
 * weights times the measure of the boundary in physical space, which the
 * assembler maps from the boundary quadrature. It neither evaluates the
 * basis nor adds into a global system. The rows and columns follow the
 * rows of the values, the functions active on the element
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class Condition
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    virtual ~Condition() = default;

    /// @brief Optional values the condition reads from ElementValues
    virtual ValueFlags flags() const noexcept = 0;

    /**
     * @brief Stiffness matrix of the condition on the boundary in the
     *        element
     *
     * @param values Values at the boundary points in the element
     * @param weights Boundary weights, one per point
     * @param stiffness Output of size (num_active, num_active),
     *        overwritten. It is resized when necessary
     *
     * @pre @p values were filled with flags(), at the points of @p weights
     */
    virtual void local_stiffness(const ElementValues<Basis, n>& values,
                                 const Eigen::VectorX<Scalar>& weights,
                                 Eigen::MatrixX<Scalar>& stiffness) const = 0;

    /**
     * @brief Load vector of the condition on the boundary in the element,
     *        from data given at its points
     *
     * @param values Values at the boundary points in the element
     * @param weights Boundary weights, one per point
     * @param data Value the condition imposes at each point
     * @param load Output of size num_active, overwritten. It is resized
     *        when necessary
     *
     * @pre @p values were filled with flags(), at the points of @p weights
     *      and @p data
     */
    virtual void local_load(const ElementValues<Basis, n>& values,
                            const Eigen::VectorX<Scalar>& weights,
                            const Eigen::VectorX<Scalar>& data,
                            Eigen::VectorX<Scalar>& load) const = 0;
};

} // namespace iguana

#endif // IGUANA_CONDITION_CONDITION_HPP
