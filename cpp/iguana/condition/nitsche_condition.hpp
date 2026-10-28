/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_NITSCHE_CONDITION_HPP
#define IGUANA_CONDITION_NITSCHE_CONDITION_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/condition/condition.hpp"
#include "iguana/element/flux.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Condition imposing values on a trace by Nitsche's method, through
 *        the flux conjugate to the trace
 *
 * @details Unlike a penalty, the method is consistent, so it imposes the
 *          values exactly for any penalty large enough to keep it stable.
 *          With a projection, the values are imposed where the data is
 *          given, as the Shifted Boundary Method needs. The adjoint term
 *          takes the sign of the symmetric method, or the opposite one of
 *          the skew-symmetric method, stable without a penalty, whose system
 *          is not symmetric
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class NitscheCondition final : public Condition<Basis, n>
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the condition imposing values on a trace
     *
     * @param trace Trace whose values are imposed
     * @param flux Flux conjugate to @p trace in the weak form of its element
     * @param penalty Penalty beta, which the size of the element scales.
     *        About 10 p^2 for a basis of degree p keeps the symmetric method
     *        stable, and any non-negative one the skew-symmetric method
     * @param symmetric Whether the method is symmetric, rather than
     *        skew-symmetric
     *
     * @throws std::invalid_argument If the penalty is negative, or zero for
     *         the symmetric method
     *
     * @pre @p trace and @p flux outlive the condition, which keeps
     *      references to them
     */
    NitscheCondition(const Trace<Basis, n>& trace,
                     const Flux<Basis, n>& flux, Scalar penalty,
                     bool symmetric = true);

    /// @brief Penalty beta
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    /// @brief Whether the method is symmetric, rather than skew-symmetric
    constexpr bool symmetric() const noexcept
    { return symmetric_; }

    ValueFlags flags() const noexcept override
    {
        return {.physical_gradients = trace_.flags().physical_gradients
                                      || flux_.flags().physical_gradients};
    }

    void local_tangent(const ElementValues<Basis, n>& values,
                       const ElementValues<Basis, n>& shifted,
                       const Eigen::VectorX<Scalar>& weights,
                       const PointMatrix<Scalar, n>& normals,
                       const Eigen::VectorX<Scalar>& data,
                       const Eigen::VectorX<Scalar>& coefficients,
                       Eigen::MatrixX<Scalar>& tangent) const override;

    void local_residual(const ElementValues<Basis, n>& values,
                        const ElementValues<Basis, n>& shifted,
                        const Eigen::VectorX<Scalar>& weights,
                        const PointMatrix<Scalar, n>& normals,
                        const Eigen::VectorX<Scalar>& data,
                        const Eigen::VectorX<Scalar>& coefficients,
                        Eigen::VectorX<Scalar>& residual) const override;

private:
    /// @brief Trace whose values are imposed
    const Trace<Basis, n>& trace_;

    /// @brief Flux conjugate to the trace
    const Flux<Basis, n>& flux_;

    /// @brief Penalty beta
    Scalar penalty_;

    /// @brief Whether the method is symmetric, rather than skew-symmetric
    bool symmetric_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_NITSCHE_CONDITION_HPP
