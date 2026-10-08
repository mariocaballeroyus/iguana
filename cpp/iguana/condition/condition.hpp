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
 * assembler maps from the boundary quadrature. It receives the values at
 * the points and the shifted ones, expanded towards where the data is
 * given, as the shifted boundary method needs; both are the same on a
 * quadrature that is not shifted. It neither evaluates the basis nor adds
 * into a global system. The rows and columns follow the rows of the
 * values, the functions active on the element
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
     * @param shifted Values where the data is given: expanded towards the
     *        closest points on the boundary of a shifted quadrature, and
     *        the values themselves on a quadrature that is not shifted
     * @param weights Boundary weights, one per point
     * @param normals Unit normals of the boundary in physical space,
     *        pointing out of the domain, of size (num_points, n)
     * @param stiffness Output of size (num_active, num_active),
     *        overwritten. It is resized when necessary
     *
     * @pre @p values and @p shifted were filled with flags(), at the points
     *      of @p weights and @p normals
     */
    virtual void local_stiffness(const ElementValues<Basis, n>& values,
                                 const ElementValues<Basis, n>& shifted,
                                 const Eigen::VectorX<Scalar>& weights,
                                 const PointMatrix<Scalar, n>& normals,
                                 Eigen::MatrixX<Scalar>& stiffness) const = 0;

    /**
     * @brief Load vector of the condition on the boundary in the element,
     *        from data given at its points
     *
     * @param values Values at the boundary points in the element
     * @param shifted Values where the data is given: expanded towards the
     *        closest points on the boundary of a shifted quadrature, and
     *        the values themselves on a quadrature that is not shifted
     * @param weights Boundary weights, one per point
     * @param normals Unit normals of the boundary in physical space,
     *        pointing out of the domain, of size (num_points, n)
     * @param data Data of the condition at each point, such as the value
     *        it imposes or a flux
     * @param load Output of size num_active, overwritten. It is resized
     *        when necessary
     *
     * @pre @p values and @p shifted were filled with flags(), at the points
     *      of @p weights, @p normals and @p data
     */
    virtual void local_load(const ElementValues<Basis, n>& values,
                            const ElementValues<Basis, n>& shifted,
                            const Eigen::VectorX<Scalar>& weights,
                            const PointMatrix<Scalar, n>& normals,
                            const Eigen::VectorX<Scalar>& data,
                            Eigen::VectorX<Scalar>& load) const = 0;
};

} // namespace iguana

#endif // IGUANA_CONDITION_CONDITION_HPP
