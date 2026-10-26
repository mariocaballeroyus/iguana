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
 * @brief Condition on a boundary, giving its residual and tangent on the
 *        part of the boundary in one element
 *
 * @details It reads the values its flags() request, at the points and
 *          where the data is given, the same without a projection, and
 *          integrates with the boundary weights the assembler maps. Its
 *          residual and tangent are taken at the coefficients of the
 *          unknowns on the element, as for Element, and its rows and
 *          columns follow the num_dofs degrees of freedom of the element,
 *          ordered as for Element
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
     * @brief Tangent matrix of the condition on the boundary in the
     *        element, the derivative of its residual with respect to the
     *        coefficients, at the coefficients
     *
     * @param values Values at the boundary points in the element
     * @param shifted Values where the data is given: expanded towards the
     *        closest points of a projection, and the values themselves
     *        without one
     * @param weights Boundary weights, one per point
     * @param normals Unit normals of the boundary in physical space,
     *        pointing out of the domain, of size (num_points, n)
     * @param data Data of the condition at each point, such as the value
     *        it imposes or a flux
     * @param coefficients Coefficients of the unknowns on the element, one
     *        per degree of freedom
     * @param tangent Output of size (num_dofs, num_dofs), overwritten. It
     *        is resized when necessary
     *
     * @pre @p values and @p shifted were filled with flags(), at the points
     *      of @p weights, @p normals and @p data, and @p coefficients has
     *      num_dofs entries
     */
    virtual void local_tangent(const ElementValues<Basis, n>& values,
                               const ElementValues<Basis, n>& shifted,
                               const Eigen::VectorX<Scalar>& weights,
                               const PointMatrix<Scalar, n>& normals,
                               const Eigen::VectorX<Scalar>& data,
                               const Eigen::VectorX<Scalar>& coefficients,
                               Eigen::MatrixX<Scalar>& tangent) const = 0;

    /**
     * @brief Residual vector of the condition on the boundary in the
     *        element at the coefficients, its weak form tested by each
     *        function
     *
     * @param values Values at the boundary points in the element
     * @param shifted Values where the data is given, as for local_tangent()
     * @param weights Boundary weights, one per point
     * @param normals Unit normals of the boundary in physical space,
     *        pointing out of the domain, of size (num_points, n)
     * @param data Data of the condition at each point, such as the value
     *        it imposes or a flux
     * @param coefficients Coefficients of the unknowns on the element, one
     *        per degree of freedom
     * @param residual Output of size num_dofs, overwritten. It is resized
     *        when necessary
     *
     * @pre As for local_tangent()
     */
    virtual void local_residual(const ElementValues<Basis, n>& values,
                                const ElementValues<Basis, n>& shifted,
                                const Eigen::VectorX<Scalar>& weights,
                                const PointMatrix<Scalar, n>& normals,
                                const Eigen::VectorX<Scalar>& data,
                                const Eigen::VectorX<Scalar>& coefficients,
                                Eigen::VectorX<Scalar>& residual) const = 0;
};

} // namespace iguana

#endif // IGUANA_CONDITION_CONDITION_HPP
