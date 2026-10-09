/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_ELEMENT_HPP
#define IGUANA_ELEMENT_ELEMENT_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"

namespace iguana
{

/**
 * @brief Physics of a problem on one element, giving its local matrix and
 *        vector from the values at the points of the element
 *
 * An element reads from ElementValues the values its flags() request and
 * integrates with physical weights, the quadrature weights times the
 * measure of the patch. It neither evaluates the basis nor adds into a
 * global system, which the assembler does. The rows and columns follow
 * the rows of the values, the functions active on the element
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class Element
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    virtual ~Element() = default;

    /// @brief Optional values the element reads from ElementValues
    virtual ValueFlags flags() const noexcept = 0;

    /**
     * @brief Stiffness matrix of the element, the matrix of its bilinear
     *        form
     *
     * @param values Values at the points of the element
     * @param weights Physical weights, one per point, scaled by the
     *        weight of the test functions on the cell when the
     *        assembly gives one
     * @param stiffness Output of size (num_active, num_active),
     *        overwritten. It is resized when necessary
     *
     * @pre @p values were filled with flags(), at the points of @p weights
     */
    virtual void local_stiffness(const ElementValues<Basis, n>& values,
                                 const Eigen::VectorX<Scalar>& weights,
                                 Eigen::MatrixX<Scalar>& stiffness) const = 0;

    /**
     * @brief Load vector of the element, the vector of its linear form,
     *        from a source given at its points
     *
     * @param values Values at the points of the element
     * @param weights Physical weights, one per point, scaled by the
     *        weight of the test functions on the cell when the
     *        assembly gives one
     * @param source Value of the source at each point
     * @param load Output of size num_active, overwritten. It is resized
     *        when necessary
     *
     * @pre @p values were filled with flags(), at the points of @p weights
     *      and @p source
     */
    virtual void local_load(const ElementValues<Basis, n>& values,
                            const Eigen::VectorX<Scalar>& weights,
                            const Eigen::VectorX<Scalar>& source,
                            Eigen::VectorX<Scalar>& load) const = 0;

    /**
     * @brief Mass matrix of the element, the matrix of the time derivative
     *        of its unknowns
     *
     * @param values Values at the points of the element
     * @param weights Physical weights, one per point, scaled by the
     *        weight of the test functions on the cell when the
     *        assembly gives one
     * @param mass Output of size (num_active, num_active), overwritten. It
     *        is resized when necessary
     *
     * @pre @p values were filled with flags(), at the points of @p weights
     */
    virtual void local_mass(const ElementValues<Basis, n>& values,
                            const Eigen::VectorX<Scalar>& weights,
                            Eigen::MatrixX<Scalar>& mass) const = 0;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_ELEMENT_HPP
