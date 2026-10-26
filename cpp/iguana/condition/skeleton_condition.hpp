/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_SKELETON_CONDITION_HPP
#define IGUANA_CONDITION_SKELETON_CONDITION_HPP

#include <array>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"

namespace iguana
{

/**
 * @brief Condition on the interior faces of a grid, giving its residual and
 *        tangent on one face from the two cells on either side
 *
 * @details The two-sided sibling of Condition: each cell gives its one-sided
 *          values at the face and a weight for its test functions, such as
 *          its volume fraction in the Generalized Shifted Boundary Method.
 *          Rows and columns follow the degrees of freedom of the cell before
 *          the face, then those of the cell after, each ordered as for
 *          Element, and so do the coefficients, a function of both cells
 *          having a row in each
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class SkeletonCondition
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    virtual ~SkeletonCondition() = default;

    /// @brief Optional values the condition reads from ElementValues
    virtual ValueFlags flags() const noexcept = 0;

    /**
     * @brief Tangent matrix of the condition on one face, the derivative of
     *        its residual with respect to the coefficients, at the
     *        coefficients
     *
     * @param before Values of the cell before the face at its points
     * @param after Values of the cell after the face at the same points
     * @param shifted_before Values of the cell before the face where the
     *        data is given: expanded towards the closest points of a
     *        projection, and the values themselves without one
     * @param shifted_after Values of the cell after the face where the data
     *        is given, as for @p shifted_before
     * @param cell_weights Weights of the test functions on the cell before
     *        the face and on the cell after it
     * @param weights Face weights, one per point
     * @param normals Unit normals of the face in physical space, pointing
     *        from the cell before it to the cell after it, of size
     *        (num_points, n)
     * @param data Data of the condition at each point, such as the value it
     *        imposes, given at its closest point with a projection
     * @param coefficients Coefficients of the unknowns on both cells
     *        together, one per row
     * @param tangent Output of size (num_dofs, num_dofs) of both cells
     *        together, overwritten. It is resized when necessary
     *
     * @pre The values were filled with flags(), at the points of @p weights,
     *      @p normals and @p data, and @p coefficients has num_dofs entries
     */
    virtual void local_tangent(
        const ElementValues<Basis, n>& before,
        const ElementValues<Basis, n>& after,
        const ElementValues<Basis, n>& shifted_before,
        const ElementValues<Basis, n>& shifted_after,
        const std::array<Scalar, 2>& cell_weights,
        const Eigen::VectorX<Scalar>& weights,
        const PointMatrix<Scalar, n>& normals,
        const Eigen::VectorX<Scalar>& data,
        const Eigen::VectorX<Scalar>& coefficients,
        Eigen::MatrixX<Scalar>& tangent) const = 0;

    /**
     * @brief Residual vector of the condition on one face at the
     *        coefficients, its weak form tested by each function
     *
     * @param before Values of the cell before the face at its points
     * @param after Values of the cell after the face at the same points
     * @param shifted_before Values of the cell before the face where the
     *        data is given, as for local_tangent()
     * @param shifted_after Values of the cell after the face where the data
     *        is given, as for local_tangent()
     * @param cell_weights Weights of the test functions on the cell before
     *        the face and on the cell after it
     * @param weights Face weights, one per point
     * @param normals Unit normals of the face in physical space, pointing
     *        from the cell before it to the cell after it, of size
     *        (num_points, n)
     * @param data Data of the condition at each point, such as the value it
     *        imposes, given at its closest point with a projection
     * @param coefficients Coefficients of the unknowns on both cells
     *        together, one per row
     * @param residual Output of size num_dofs of both cells together,
     *        overwritten. It is resized when necessary
     *
     * @pre As for local_tangent()
     */
    virtual void local_residual(
        const ElementValues<Basis, n>& before,
        const ElementValues<Basis, n>& after,
        const ElementValues<Basis, n>& shifted_before,
        const ElementValues<Basis, n>& shifted_after,
        const std::array<Scalar, 2>& cell_weights,
        const Eigen::VectorX<Scalar>& weights,
        const PointMatrix<Scalar, n>& normals,
        const Eigen::VectorX<Scalar>& data,
        const Eigen::VectorX<Scalar>& coefficients,
        Eigen::VectorX<Scalar>& residual) const = 0;
};

} // namespace iguana

#endif // IGUANA_CONDITION_SKELETON_CONDITION_HPP
