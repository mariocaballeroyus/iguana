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
 * @brief Condition imposing values g on a trace of an element by Nitsche's
 *        method, through the flux conjugate to the trace
 *
 * Its stiffness keeps the boundary term of the weak form, which pairs the
 * flux F with the trace T, adds its adjoint and a penalty scaled by the size
 * h of the element:
 *
 *     K_ij = - integral of F(N_j) T(N_i) ds - integral of F(N_i) T(S N_j) ds
 *            + gamma integral of T(S N_i) T(S N_j) / h ds
 *
 * Its load is F_i = integral of (gamma T(S N_i) / h - F(N_i)) g ds. The
 * boundary term holds at the points, while S N reaches where the data is
 * given, the closest points of a shifted quadrature as the shifted boundary
 * method needs, and N itself otherwise. Unlike a penalty, the method is
 * consistent, so it imposes the values exactly for any penalty large enough
 * to keep it stable
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
     * @param penalty Penalty gamma, which the size of the element scales.
     *        About 10 p^2 for a basis of degree p keeps the method stable
     *
     * @throws std::invalid_argument If the penalty is not positive
     *
     * @pre @p trace and @p flux outlive the condition, which keeps
     *      references to them
     */
    NitscheCondition(const Trace<Basis, n>& trace,
                     const Flux<Basis, n>& flux, Scalar penalty);

    /// @brief Penalty gamma
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

    /// @brief Penalty gamma
    Scalar penalty_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_NITSCHE_CONDITION_HPP
