/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_VOLUME_GHOST_PENALTY_HPP
#define IGUANA_CONDITION_VOLUME_GHOST_PENALTY_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/trace.hpp"

namespace iguana
{

/**
 * @brief Ghost penalty extending the field of a trace smoothly across the
 *        interior faces of a grid, through the polynomials of the cells
 *
 * @details On each face it penalizes, over both of its cells, the difference
 *          between the polynomial of a cell and that of the other one
 *          extended into it. It vanishes on a field that is one polynomial
 *          across the face, so the penalty is consistent, and needs no
 *          derivatives, so it holds for any continuity and map. For splines
 *          of maximal continuity on equal cells it is SkeletonGhostPenalty
 *          with the penalty scaled by 2 / ((p!)^2 (2p + 1)), the factors the
 *          polynomials give once integrated. Its sign is that of the block
 *          of its field in the system, as for SkeletonGhostPenalty. Rows and
 *          columns follow the degrees of freedom of the cell before the
 *          face, then those of the cell after, each ordered as for Element
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class VolumeGhostPenalty final
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the penalty on a trace
     *
     *     γ h^(s − 2)
     *
     * @details The penalty over the cells of a face, with h the mean size
     *          of its two cells and s the scaling
     *
     * @param trace Trace whose polynomials are compared
     * @param penalty Penalty γ, of the sign of the block of the field of
     *        @p trace
     * @param scaling Extra powers s of the size of the cells, two for the
     *        pressure of StokesStabilizedElement
     *
     * @throws std::invalid_argument If the penalty is zero, or if the
     *         scaling is negative
     *
     * @pre @p trace outlives the penalty, which keeps a reference to it
     */
    VolumeGhostPenalty(const Trace<Basis, n>& trace, Scalar penalty,
                       int scaling = 0);

    /// @brief Penalty γ
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    /// @brief Extra powers s of the size of the cells
    constexpr int scaling() const noexcept
    { return scaling_; }

    /// @brief Optional values the penalty reads from ElementValues
    ValueFlags flags() const noexcept
    { return trace_.flags(); }

    /**
     * @brief Field the penalty extends, that of its trace
     *
     * @param num_fields Number of fields of the space
     */
    std::size_t field(std::size_t num_fields) const noexcept
    { return trace_.field(num_fields); }

    /**
     * @brief Tangent matrix of the penalty on one face, the same at any
     *        coefficients
     *
     * @param before Values of the cell before the face at its own points
     * @param after_in_before Values of the cell after the face, its
     *        polynomial extended, at the points of the cell before
     * @param before_in_after Values of the cell before the face, its
     *        polynomial extended, at the points of the cell after
     * @param after Values of the cell after the face at its own points
     * @param weights_before Volume weights of the cell before, one per point
     * @param weights_after Volume weights of the cell after, one per point
     * @param coefficients Coefficients of the unknowns on both cells
     *        together, one per row, as for SkeletonCondition
     * @param tangent Output of size (num_dofs, num_dofs) of both cells
     *        together, overwritten. It is resized when necessary
     *
     * @pre The values were filled with flags() at the points of the cell
     *      the weights of the same side belong to, and @p coefficients has
     *      num_dofs entries
     */
    void local_tangent(const ElementValues<Basis, n>& before,
                       const ElementValues<Basis, n>& after_in_before,
                       const ElementValues<Basis, n>& before_in_after,
                       const ElementValues<Basis, n>& after,
                       const Eigen::VectorX<Scalar>& weights_before,
                       const Eigen::VectorX<Scalar>& weights_after,
                       const Eigen::VectorX<Scalar>& coefficients,
                       Eigen::MatrixX<Scalar>& tangent) const;

    /**
     * @brief Residual vector of the penalty on one face at the coefficients,
     *        its tangent times them
     *
     * @param before Values of the cell before the face at its own points
     * @param after_in_before Values of the cell after the face, its
     *        polynomial extended, at the points of the cell before
     * @param before_in_after Values of the cell before the face, its
     *        polynomial extended, at the points of the cell after
     * @param after Values of the cell after the face at its own points
     * @param weights_before Volume weights of the cell before, one per point
     * @param weights_after Volume weights of the cell after, one per point
     * @param coefficients Coefficients of the unknowns on both cells
     *        together, one per row, as for SkeletonCondition
     * @param residual Output of size num_dofs of both cells together,
     *        overwritten. It is resized when necessary
     *
     * @pre As for local_tangent()
     */
    void local_residual(const ElementValues<Basis, n>& before,
                        const ElementValues<Basis, n>& after_in_before,
                        const ElementValues<Basis, n>& before_in_after,
                        const ElementValues<Basis, n>& after,
                        const Eigen::VectorX<Scalar>& weights_before,
                        const Eigen::VectorX<Scalar>& weights_after,
                        const Eigen::VectorX<Scalar>& coefficients,
                        Eigen::VectorX<Scalar>& residual) const;

private:
    /// @brief Trace whose polynomials are compared
    const Trace<Basis, n>& trace_;

    /// @brief Penalty γ
    Scalar penalty_;

    /// @brief Extra powers s of the size of the cells
    int scaling_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_VOLUME_GHOST_PENALTY_HPP
