/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_SKELETON_GHOST_PENALTY_HPP
#define IGUANA_CONDITION_SKELETON_GHOST_PENALTY_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/trace.hpp"

namespace iguana
{

/**
 * @brief Ghost penalty extending the field of a trace smoothly across the
 *        interior faces of a grid
 *
 * @details It penalizes the jumps of the derivatives of order p, the degree
 *          across a face, the only ones that jump for splines of maximal
 *          continuity. They vanish on a field that is one polynomial across
 *          the faces, so the penalty is consistent. Its sign is that of
 *          the block of its field in the system, which it keeps definite:
 *          positive for the field of PoissonElement or the velocity of
 *          StokesElement, negative for the pressure of StokesElement, whose
 *          block is negative in the symmetric form. Rows and columns follow
 *          the degrees of freedom of the cell before the face, then those of
 *          the cell after, each ordered as for Element
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class SkeletonGhostPenalty final
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the penalty on a trace
     *
     *     β h^(2p − 1 + s)
     *
     * @details The penalty on a face, with h the mean size of its two
     *          cells, p the degree across it and s the scaling
     *
     * @param trace Trace whose jumps are penalized
     * @param penalty Penalty β, of the sign of the block of the field of
     *        @p trace
     * @param scaling Extra powers s of the size of the cells, two for the
     *        pressure of StokesElement
     *
     * @throws std::invalid_argument If the penalty is zero, or if the
     *         scaling is negative
     *
     * @pre @p trace reads the values alone, as the field of PoissonElement
     *      does, so that it differentiates with them, and it outlives the
     *      penalty, which keeps a reference to it
     */
    SkeletonGhostPenalty(const Trace<Basis, n>& trace, Scalar penalty,
                         int scaling = 0);

    /// @brief Penalty β
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    /// @brief Extra powers s of the size of the cells
    constexpr int scaling() const noexcept
    { return scaling_; }

    /// @brief Optional values the penalty reads from ElementValues
    ValueFlags flags() const noexcept
    { return trace_.flags(); }

    /**
     * @brief Stiffness matrix of the penalty on one face
     *
     * @param before Values of the cell before the face, differentiated
     *        across it at the face points
     * @param after Values of the cell after the face, differentiated across
     *        it at the same points
     * @param weights Face weights, one per point
     * @param order Order p of the derivatives the values hold
     * @param stiffness Output of size (num_dofs, num_dofs) of both
     *        cells together, overwritten. It is resized when necessary
     *
     * @pre @p before and @p after were filled with flags() at the points of
     *      @p weights and differentiated across the face
     */
    void local_stiffness(const ElementValues<Basis, n>& before,
                         const ElementValues<Basis, n>& after,
                         const Eigen::VectorX<Scalar>& weights, int order,
                         Eigen::MatrixX<Scalar>& stiffness) const;

private:
    /// @brief Trace whose jumps are penalized
    const Trace<Basis, n>& trace_;

    /// @brief Penalty β
    Scalar penalty_;

    /// @brief Extra powers s of the size of the cells
    int scaling_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_SKELETON_GHOST_PENALTY_HPP
