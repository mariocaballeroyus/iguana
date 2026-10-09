/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_FACE_NITSCHE_CONDITION_HPP
#define IGUANA_CONDITION_FACE_NITSCHE_CONDITION_HPP

#include <array>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/condition/face_condition.hpp"
#include "iguana/element/flux.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Condition imposing values g on a trace of an element by Nitsche's
 *        method across the faces where the weights of the test functions
 *        jump, as the generalized shifted boundary method does
 *
 * With alpha_b and alpha_a the weights of the cells before and after a
 * face, j = alpha_b - alpha_a their jump along the normal n from the cell
 * before to the cell after, and {{w}} = (alpha_b w_b + alpha_a w_a) /
 * (alpha_b + alpha_a) the mean of a quantity of the two cells weighted by
 * them, its stiffness is
 *
 *     K_ij = - j integral of T(N_i) {{F(N_j)}} ds
 *            - j integral of {{F(N_i)}} {{T(S N_j)}} ds
 *            + beta |j| integral of {{T(S N_i)}} {{T(S N_j)}} / h ds
 *
 * and its load F_i = integral of (beta |j| {{T(S N_i)}} / h
 * - j {{F(N_i)}}) g ds, with h = {{h}}. The first term is the jump of the
 * weighted test functions, whose traces are continuous, against the flux
 * F along n. The other two impose the values where the data is given,
 * S N expanding each cell's own functions to the closest points of a
 * shifted quadrature. With weights of one inside and zero outside, the
 * condition is NitscheCondition on the surrogate boundary between two
 * cells
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class FaceNitscheCondition final : public FaceCondition<Basis, n>
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the condition imposing values on a trace
     *
     * @param trace Trace whose values are imposed
     * @param flux Flux conjugate to @p trace in the weak form of its element
     * @param penalty Penalty beta, which the size of the cells scales, as
     *        for NitscheCondition
     *
     * @throws std::invalid_argument If the penalty is not positive
     *
     * @pre @p trace and @p flux outlive the condition, which keeps
     *      references to them
     */
    FaceNitscheCondition(const Trace<Basis, n>& trace,
                         const Flux<Basis, n>& flux, Scalar penalty);

    /// @brief Penalty beta
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    ValueFlags flags() const noexcept override
    {
        return {.physical_gradients = trace_.flags().physical_gradients
                                      || flux_.flags().physical_gradients};
    }

    /// @pre The cell weights are not both zero
    void local_stiffness(const ElementValues<Basis, n>& before,
                         const ElementValues<Basis, n>& after,
                         const ElementValues<Basis, n>& shifted_before,
                         const ElementValues<Basis, n>& shifted_after,
                         const std::array<Scalar, 2>& cell_weights,
                         const Eigen::VectorX<Scalar>& weights,
                         const PointMatrix<Scalar, n>& normals,
                         Eigen::MatrixX<Scalar>& stiffness) const override;

    /// @pre The cell weights are not both zero
    void local_load(const ElementValues<Basis, n>& before,
                    const ElementValues<Basis, n>& after,
                    const ElementValues<Basis, n>& shifted_before,
                    const ElementValues<Basis, n>& shifted_after,
                    const std::array<Scalar, 2>& cell_weights,
                    const Eigen::VectorX<Scalar>& weights,
                    const PointMatrix<Scalar, n>& normals,
                    const Eigen::VectorX<Scalar>& data,
                    Eigen::VectorX<Scalar>& load) const override;

private:
    /// @brief Trace whose values are imposed
    const Trace<Basis, n>& trace_;

    /// @brief Flux conjugate to the trace
    const Flux<Basis, n>& flux_;

    /// @brief Penalty beta
    Scalar penalty_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_FACE_NITSCHE_CONDITION_HPP
