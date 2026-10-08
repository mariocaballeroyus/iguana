/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_STABILIZATION_GHOST_PENALTY_HPP
#define IGUANA_STABILIZATION_GHOST_PENALTY_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/trace.hpp"

namespace iguana
{

/**
 * @brief Ghost penalty on a trace of an element, which extends its field
 *        smoothly across interior faces of a grid
 *
 * Its stiffness penalizes the jump across each face of the derivative of
 * order p of the trace along the normal, K_ij = gamma integral of
 * h^(2p - 1) [[d_n^p T(N_i)]] [[d_n^p T(N_j)]] ds, with p the degree across
 * the face, the only order that jumps for splines of maximal continuity.
 * It vanishes on any field that is one polynomial across the faces, so it
 * is consistent, and where the physical domain leaves the field
 * undetermined, it determines it as the extension of its neighbors
 *
 * It reads the values of the two cells of a face differentiated across it
 * by ElementValues::differentiate(), so that the trace gives the derivative
 * of the trace. The rows and columns follow the functions active on the
 * cell before the face, then those active on the cell after it
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class GhostPenalty final
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the penalty on a trace
     *
     * @param trace Trace whose jumps are penalized
     * @param penalty Penalty gamma, which the size of the elements scales
     *
     * @throws std::invalid_argument If the penalty is not positive
     *
     * @pre @p trace reads the values alone, as the field of PoissonElement
     *      does, so that it differentiates with them, and it outlives the
     *      penalty, which keeps a reference to it
     */
    GhostPenalty(const Trace<Basis, n>& trace, Scalar penalty);

    /// @brief Penalty gamma
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

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
     * @param stiffness Output of size (num_active, num_active) of both
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

    /// @brief Penalty gamma
    Scalar penalty_;
};

} // namespace iguana

#endif // IGUANA_STABILIZATION_GHOST_PENALTY_HPP
