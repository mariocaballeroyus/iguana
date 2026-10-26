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
    const ElementValues<Basis, n>& values, const ElementValues<Basis, n>&,
    const Eigen::VectorX<Scalar>&, const PointMatrix<Scalar, n>&,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    // The data is known, so the condition adds no stiffness, over every
    // degree of freedom of the element as the rows of the trace
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(values, trace);
    stiffness.setZero(trace.rows(), trace.rows());
}

template<typename Basis, std::size_t n>
void NeumannCondition<Basis, n>::local_load(
    const ElementValues<Basis, n>& values, const ElementValues<Basis, n>&,
    const Eigen::VectorX<Scalar>& weights, const PointMatrix<Scalar, n>&,
    const Eigen::VectorX<Scalar>& data, Eigen::VectorX<Scalar>& load) const
{
    // T(N_i), traces of the test functions the data does work on
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(values, trace);

    // F_i = ( T(N_i), g )_Γ
    //     ≈ Σ_q w_q g(x_q) B_iq
    load = trace * weights.cwiseProduct(data);
}

template class NeumannCondition<TensorBSpline<double, 2>, 2>;
template class NeumannCondition<TensorBSpline<double, 2>, 3>;
template class NeumannCondition<TensorBSpline<double, 3>, 3>;
template class NeumannCondition<TensorNURBS<double, 2>, 2>;
template class NeumannCondition<TensorNURBS<double, 2>, 3>;
template class NeumannCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
