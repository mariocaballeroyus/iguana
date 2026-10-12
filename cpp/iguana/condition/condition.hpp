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
 *        part of the boundary in one element
 *
 * @details It reads the values its flags() request, at the points and
 *          where the data is given, the same without a projection, and
 *          integrates with the boundary weights the assembler maps. Its rows
 *          and columns follow the functions active on the element
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
     *        closest points of a projection, and the values themselves
     *        without one
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
     *        closest points of a projection, and the values themselves
     *        without one
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
