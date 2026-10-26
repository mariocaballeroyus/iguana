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
 * @brief Physics of a problem on one element, giving its residual, its
 *        tangent and its mass there
 *
 * @details The residual and the tangent are taken at the coefficients of
 *          the unknowns on the element, so that Newton's method drives the
 *          residual to zero with the tangent. A linear element has its
 *          stiffness as its tangent, and the stiffness times the
 *          coefficients less its load as its residual. It reads the values
 *          its flags() request, at the points of the element, and
 *          integrates with the physical weights the assembler maps. Its rows
 *          and columns follow the num_dofs degrees of freedom of the
 *          element, as MixedFunctionSpace::dofs_on_element() orders them:
 *          one field after another, each with one component after another
 *          over the active functions of its basis, as many as
 *          values(field).rows(). The number of components of a field is its
 *          block size, and a space of one field orders them as
 *          FunctionSpace::dofs_on_element()
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
     * @brief Tangent matrix of the element, the derivative of its residual
     *        with respect to the coefficients, at the coefficients
     *
     *     J_ij = ∂R_i / ∂c_j
     *
     * @param values Values at the points of the element
     * @param weights Physical weights, one per point, scaled by the
     *        weight of the test functions on the cell when the
     *        assembly gives one
     * @param source Value of the source at each point, a row per point
     *        and a column per component of the source
     * @param coefficients Coefficients of the unknowns on the element, one
     *        per degree of freedom
     * @param tangent Output of size (num_dofs, num_dofs), overwritten. It
     *        is resized when necessary
     *
     * @pre @p values were filled with flags(), at the points of @p weights
     *      and @p source, which has the components the element reads, and
     *      @p coefficients has num_dofs entries
     */
    virtual void local_tangent(const ElementValues<Basis, n>& values,
                               const Eigen::VectorX<Scalar>& weights,
                               const Eigen::MatrixX<Scalar>& source,
                               const Eigen::VectorX<Scalar>& coefficients,
                               Eigen::MatrixX<Scalar>& tangent) const = 0;

    /**
     * @brief Residual vector of the element at the coefficients, its weak
     *        form tested by each function
     *
     * @param values Values at the points of the element
     * @param weights Physical weights, one per point, scaled by the
     *        weight of the test functions on the cell when the
     *        assembly gives one
     * @param source Value of the source at each point, a row per point
     *        and a column per component of the source
     * @param coefficients Coefficients of the unknowns on the element, one
     *        per degree of freedom
     * @param residual Output of size num_dofs, overwritten. It is resized
     *        when necessary
     *
     * @pre As for local_tangent()
     */
    virtual void local_residual(const ElementValues<Basis, n>& values,
                                const Eigen::VectorX<Scalar>& weights,
                                const Eigen::MatrixX<Scalar>& source,
                                const Eigen::VectorX<Scalar>& coefficients,
                                Eigen::VectorX<Scalar>& residual) const = 0;

    /**
     * @brief Mass matrix of the element, the matrix of the time derivative
     *        of its unknowns
     *
     * @param values Values at the points of the element
     * @param weights Physical weights, one per point, scaled by the
     *        weight of the test functions on the cell when the
     *        assembly gives one
     * @param mass Output of size (num_dofs, num_dofs), overwritten. It
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
