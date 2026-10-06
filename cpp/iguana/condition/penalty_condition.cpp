/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "penalty_condition.hpp"

#include <cstddef>
#include <stdexcept>

namespace iguana
{

template<typename Basis, std::size_t n>
PenaltyCondition<Basis, n>::PenaltyCondition(const Trace<Basis, n>& trace,
                                             Scalar penalty)
    : trace_(trace), penalty_(penalty)
{
    if (penalty <= 0)
        throw std::invalid_argument("PenaltyCondition: "
                                    "the penalty must be positive");
}

template<typename Basis, std::size_t n>
void PenaltyCondition<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(values, trace);

    // K_ij = β ∫ T(N_i) T(N_j) ds
    //      ≈ Σ_q β w_q B_iq B_jq
    stiffness = penalty_ * trace * weights.asDiagonal() * trace.transpose();
}

template<typename Basis, std::size_t n>
void PenaltyCondition<Basis, n>::local_load(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::VectorX<Scalar>& data,
    Eigen::VectorX<Scalar>& load) const
{
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(values, trace);

    // F_i = β ∫ g T(N_i) ds
    //     ≈ Σ_q β w_q g(x_q) B_iq
    load = penalty_ * trace * weights.cwiseProduct(data);
}

template class PenaltyCondition<TensorBSpline<double, 2>, 2>;
template class PenaltyCondition<TensorBSpline<double, 2>, 3>;
template class PenaltyCondition<TensorBSpline<double, 3>, 3>;
template class PenaltyCondition<TensorNURBS<double, 2>, 2>;
template class PenaltyCondition<TensorNURBS<double, 2>, 3>;
template class PenaltyCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
