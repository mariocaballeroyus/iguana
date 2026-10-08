/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_NEUMANN_CONDITION_HPP
#define IGUANA_CONDITION_NEUMANN_CONDITION_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/condition/condition.hpp"
#include "iguana/element/trace.hpp"

namespace iguana
{

/**
 * @brief Condition loading a trace of an element with natural data h, such
 *        as a flux or a traction
 *
 * The data is the boundary term of the weak form where it is known, so the
 * condition adds a load and no stiffness, F_i = integral of h T(N_i) ds,
 * through the matrix B of the trace of the test functions it does work on.
 * With the field trace U of -div(grad u) = f, h is the outward flux du/dn.
 * Data that depends on the normal is given at the points
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class NeumannCondition final : public Condition<Basis, n>
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the condition loading a trace
     *
     * @param trace Trace of the test functions the data does work on
     *
     * @pre @p trace outlives the condition, which keeps a reference to it
     */
    explicit NeumannCondition(const Trace<Basis, n>& trace);

    ValueFlags flags() const noexcept override
    { return trace_.flags(); }

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
    /// @brief Trace of the test functions the data does work on
    const Trace<Basis, n>& trace_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_NEUMANN_CONDITION_HPP
