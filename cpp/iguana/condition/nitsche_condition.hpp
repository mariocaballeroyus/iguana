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
 *          given, as the Shifted Boundary Method needs
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
     *        About 10 p^2 for a basis of degree p keeps the method stable
     *
     * @throws std::invalid_argument If the penalty is not positive
     *
     * @pre @p trace and @p flux outlive the condition, which keeps
     *      references to them
     */
    NitscheCondition(const Trace<Basis, n>& trace,
                     const Flux<Basis, n>& flux, Scalar penalty);

    /// @brief Penalty beta
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    ValueFlags flags() const noexcept override
    {
        return {.physical_gradients = trace_.flags().physical_gradients
                                      || flux_.flags().physical_gradients};
    }

    void local_stiffness(const ElementValues<Basis, n>& values,
                         const ElementValues<Basis, n>& shifted,
                         const Eigen::VectorX<Scalar>& weights,
                         const PointMatrix<Scalar, n>& normals,
                         Eigen::MatrixX<Scalar>& stiffness) const override;

    void local_load(const ElementValues<Basis, n>& values,
                    const ElementValues<Basis, n>& shifted,
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

#endif // IGUANA_CONDITION_NITSCHE_CONDITION_HPP
