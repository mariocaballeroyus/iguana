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
void PenaltyCondition<Basis, n>::local_tangent(
    const ElementValues<Basis, n>&, const ElementValues<Basis, n>& shifted,
    const Eigen::VectorX<Scalar>& weights, const PointMatrix<Scalar, n>&,
    const Eigen::VectorX<Scalar>&, const Eigen::VectorX<Scalar>&,
    Eigen::MatrixX<Scalar>& tangent) const
{
    // T(S N_i), traces where the data is given
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(shifted, trace);

    // K_ij = β ( T(S N_i), T(S N_j) )_Γ
    //      ≈ Σ_q β w_q B_iq B_jq
    tangent = penalty_ * trace * weights.asDiagonal() * trace.transpose();
}

template<typename Basis, std::size_t n>
void PenaltyCondition<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const ElementValues<Basis, n>& shifted,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Eigen::MatrixX<Scalar> tangent;
    local_tangent(values, shifted, weights, normals, data, coefficients,
                  tangent);

    // T(S N_i), traces where the data is given
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(shifted, trace);

    // F_i = β ( T(S N_i), g )_Γ
    //     ≈ Σ_q β w_q g(x_q) B_iq
    const Eigen::VectorX<Scalar> load =
        penalty_ * trace * weights.cwiseProduct(data);

    // R_i = Σ_j K_ij c_j − F_i
    residual = tangent * coefficients - load;
}

template class PenaltyCondition<TensorBSpline<double, 2>, 2>;
template class PenaltyCondition<TensorBSpline<double, 2>, 3>;
template class PenaltyCondition<TensorBSpline<double, 3>, 3>;
template class PenaltyCondition<TensorNURBS<double, 2>, 2>;
template class PenaltyCondition<TensorNURBS<double, 2>, 3>;
template class PenaltyCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
