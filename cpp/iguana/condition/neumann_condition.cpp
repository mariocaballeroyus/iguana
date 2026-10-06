/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "neumann_condition.hpp"

#include <cstddef>

namespace iguana
{

template<typename Basis, std::size_t n>
NeumannCondition<Basis, n>::NeumannCondition(const Trace<Basis, n>& trace)
    : trace_(trace)
{
}

template<typename Basis, std::size_t n>
void NeumannCondition<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& values, const Eigen::VectorX<Scalar>&,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    // The data is known, so the condition adds no stiffness
    const Eigen::Index num_active = values.values().rows();
    stiffness.setZero(num_active, num_active);
}

template<typename Basis, std::size_t n>
void NeumannCondition<Basis, n>::local_load(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::VectorX<Scalar>& data,
    Eigen::VectorX<Scalar>& load) const
{
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(values, trace);

    // F_i = ∫ h T(N_i) ds
    //     ≈ Σ_q w_q h(x_q) B_iq
    load = trace * weights.cwiseProduct(data);
}

template class NeumannCondition<TensorBSpline<double, 2>, 2>;
template class NeumannCondition<TensorBSpline<double, 2>, 3>;
template class NeumannCondition<TensorBSpline<double, 3>, 3>;
template class NeumannCondition<TensorNURBS<double, 2>, 2>;
template class NeumannCondition<TensorNURBS<double, 2>, 3>;
template class NeumannCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
