/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_PENALTY_CONDITION_HPP
#define IGUANA_CONDITION_PENALTY_CONDITION_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/condition/condition.hpp"
#include "iguana/element/trace.hpp"

namespace iguana
{

/**
 * @brief Condition imposing values on a trace through a penalty
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class PenaltyCondition final : public Condition<Basis, n>
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the condition imposing values on a trace
     *
     * @param trace Trace whose values are imposed
     * @param penalty Penalty beta, which weighs the imposed values against
     *        the rest of the problem
     *
     * @throws std::invalid_argument If the penalty is not positive
     *
     * @pre @p trace outlives the condition, which keeps a reference to it
     */
    PenaltyCondition(const Trace<Basis, n>& trace, Scalar penalty);

    /// @brief Penalty beta
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    ValueFlags flags() const noexcept override
    { return trace_.flags(); }

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

    /// @brief Penalty beta
    Scalar penalty_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_PENALTY_CONDITION_HPP
