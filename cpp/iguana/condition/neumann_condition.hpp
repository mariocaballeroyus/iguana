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
 * @brief Condition loading a trace with natural data, such as a flux or a
 *        traction
 *
 * @details The data is the boundary term of the weak form where it is
 *          known, so the condition adds a load and no tangent. With the
 *          field trace of −div(grad u) = f, the data is the outward flux
 *          du/dn
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
    /// @brief Trace of the test functions the data does work on
    const Trace<Basis, n>& trace_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_NEUMANN_CONDITION_HPP
