/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_SKELETON_MIXED_GHOST_PENALTY_HPP
#define IGUANA_CONDITION_SKELETON_MIXED_GHOST_PENALTY_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"

namespace iguana
{

/**
 * @brief Mixed ghost penalty, pairing the divergence of the velocity of a
 *        fluid with its pressure across the interior faces of a grid
 *
 *     − s(div u, q) − s(div v, p)
 *     s(div u, q) = Σ_F β h^(2k + 1) ( [[∂ₙᵏ div u]], [[∂ₙᵏ q]] )_F
 *
 * @details Beside the continuity equations weighted by the volume
 *          fractions, it extends the pressure beyond the fluid while testing
 *          the pressure functions with the divergence alone. On a
 *          divergence-conforming space the divergence lies in the pressure
 *          space, so the continuity equations still make it vanish in every
 *          cell, those beyond the fluid included, which a ghost penalty on
 *          the pressure itself would break. Of the derivatives of every order
 *          up to k, it keeps those of order k, the degree of the pressure
 *          across the face, the only ones that jump for splines of maximal
 *          continuity. Rows and columns follow the degrees of freedom of the
 *          cell before the face, then those of the cell after, each ordered
 *          field after field: the components of the velocity, then the
 *          pressure
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class SkeletonMixedGhostPenalty final
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the penalty
     *
     * @param penalty Penalty β
     *
     * @throws std::invalid_argument If the penalty is not positive
     */
    explicit SkeletonMixedGhostPenalty(Scalar penalty);

    /// @brief Penalty β
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    /// @brief Optional values the penalty reads from ElementValues
    ValueFlags flags() const noexcept
    { return {}; }

    /**
     * @brief Tangent matrix of the penalty on one face, the same at any
     *        coefficients
     *
     * @param before Values of the cell before the face, each component of
     *        the velocity derived once along its own direction and k times
     *        along the normal of the face, and the pressure k times along
     *        that normal, in physical space
     * @param after Values of the cell after the face, derived alike at the
     *        same points
     * @param weights Face weights, one per point
     * @param order Order k of the derivatives of the pressure
     * @param coefficients Coefficients of the unknowns on both cells
     *        together, one per row
     * @param tangent Output of size (num_dofs, num_dofs) of both cells
     *        together, overwritten. It is resized when necessary
     *
     * @pre @p before and @p after have the n components of the velocity and
     *      the pressure as their fields, in that order, derived as above
     */
    void local_tangent(const ElementValues<Basis, n>& before,
                       const ElementValues<Basis, n>& after,
                       const Eigen::VectorX<Scalar>& weights, int order,
                       const Eigen::VectorX<Scalar>& coefficients,
                       Eigen::MatrixX<Scalar>& tangent) const;

    /**
     * @brief Residual vector of the penalty on one face at the coefficients,
     *        its tangent times them
     *
     * @param before Values of the cell before the face, derived as for
     *        local_tangent()
     * @param after Values of the cell after the face, derived alike
     * @param weights Face weights, one per point
     * @param order Order k of the derivatives of the pressure
     * @param coefficients Coefficients of the unknowns on both cells
     *        together, one per row
     * @param residual Output of size num_dofs of both cells together,
     *        overwritten. It is resized when necessary
     *
     * @pre As for local_tangent()
     */
    void local_residual(const ElementValues<Basis, n>& before,
                        const ElementValues<Basis, n>& after,
                        const Eigen::VectorX<Scalar>& weights, int order,
                        const Eigen::VectorX<Scalar>& coefficients,
                        Eigen::VectorX<Scalar>& residual) const;

private:
    /// @brief Penalty β
    Scalar penalty_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_SKELETON_MIXED_GHOST_PENALTY_HPP
