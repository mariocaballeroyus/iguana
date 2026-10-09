/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_FACE_CONDITION_HPP
#define IGUANA_CONDITION_FACE_CONDITION_HPP

#include <array>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"

namespace iguana
{

/**
 * @brief Condition on interior faces of a grid, giving its stiffness and
 *        load on one face from the values of the two cells on either side
 *
 * The two-sided sibling of Condition: on a face between a cell before it
 * and a cell after it, along its direction, the condition reads the values
 * of each cell at the face points, each with the polynomial of its own
 * cell, and their shifted values, expanded towards where the data is
 * given. It also receives the weight of the test functions on each cell,
 * such as the volume fractions whose jump weights the conditions of the
 * generalized shifted boundary method. It integrates with face weights,
 * the quadrature weights times the measure of the face in physical space,
 * which the assembler maps. The rows and columns follow the functions
 * active on the cell before the face, then those active on the cell after
 * it, so that a function active on both gets the sum of its two rows
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class FaceCondition
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    virtual ~FaceCondition() = default;

    /// @brief Optional values the condition reads from ElementValues
    virtual ValueFlags flags() const noexcept = 0;

    /**
     * @brief Stiffness matrix of the condition on one face
     *
     * @param before Values of the cell before the face at its points
     * @param after Values of the cell after the face at the same points
     * @param shifted_before Values of the cell before the face where the
     *        data is given: expanded towards the closest points on the
     *        boundary of a shifted quadrature, and the values themselves on
     *        a quadrature that is not shifted
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
     *        imposes, given at its closest point on a shifted quadrature
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

#endif // IGUANA_CONDITION_FACE_CONDITION_HPP
