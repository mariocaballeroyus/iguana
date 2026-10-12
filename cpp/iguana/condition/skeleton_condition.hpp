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
 * @brief Condition on the interior faces of a grid, giving its stiffness
 *        and load on one face from the two cells on either side
 *
 * @details The two-sided sibling of Condition: each cell gives its one-sided
 *          values at the face and a weight for its test functions, such as
 *          its volume fraction in the Generalized Shifted Boundary Method.
 *          Rows and columns follow the functions active on the cell before
 *          the face, then those of the cell after
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
     * @brief Stiffness matrix of the condition on one face
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
     * @param stiffness Output of size (num_active, num_active) of both cells
     *        together, overwritten. It is resized when necessary
     *
     * @pre The values were filled with flags(), at the points of @p weights
     *      and @p normals
     */
    virtual void local_stiffness(
        const ElementValues<Basis, n>& before,
        const ElementValues<Basis, n>& after,
        const ElementValues<Basis, n>& shifted_before,
        const ElementValues<Basis, n>& shifted_after,
        const std::array<Scalar, 2>& cell_weights,
        const Eigen::VectorX<Scalar>& weights,
        const PointMatrix<Scalar, n>& normals,
        Eigen::MatrixX<Scalar>& stiffness) const = 0;

    /**
     * @brief Load vector of the condition on one face, from data given at
     *        its points
     *
     * @param before Values of the cell before the face at its points
     * @param after Values of the cell after the face at the same points
     * @param shifted_before Values of the cell before the face where the
     *        data is given, as for local_stiffness()
     * @param shifted_after Values of the cell after the face where the data
     *        is given, as for local_stiffness()
     * @param cell_weights Weights of the test functions on the cell before
     *        the face and on the cell after it
     * @param weights Face weights, one per point
     * @param normals Unit normals of the face in physical space, pointing
     *        from the cell before it to the cell after it, of size
     *        (num_points, n)
     * @param data Data of the condition at each point, such as the value it
     *        imposes, given at its closest point with a projection
     * @param load Output of size num_active of both cells together,
     *        overwritten. It is resized when necessary
     *
     * @pre The values were filled with flags(), at the points of @p weights,
     *      @p normals and @p data
     */
    virtual void local_load(
        const ElementValues<Basis, n>& before,
        const ElementValues<Basis, n>& after,
        const ElementValues<Basis, n>& shifted_before,
        const ElementValues<Basis, n>& shifted_after,
        const std::array<Scalar, 2>& cell_weights,
        const Eigen::VectorX<Scalar>& weights,
        const PointMatrix<Scalar, n>& normals,
        const Eigen::VectorX<Scalar>& data,
        Eigen::VectorX<Scalar>& load) const = 0;
};

} // namespace iguana

#endif // IGUANA_CONDITION_SKELETON_CONDITION_HPP
