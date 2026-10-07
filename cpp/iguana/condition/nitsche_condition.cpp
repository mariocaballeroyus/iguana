/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "nitsche_condition.hpp"

#include <cstddef>
#include <stdexcept>

namespace iguana
{

template<typename Basis, std::size_t n>
NitscheCondition<Basis, n>::NitscheCondition(const Trace<Basis, n>& trace,
                                             const Flux<Basis, n>& flux,
                                             Scalar penalty)
    : trace_(trace), flux_(flux), penalty_(penalty)
{
    if (penalty <= 0)
        throw std::invalid_argument("NitscheCondition: "
                                    "the penalty must be positive");
}

template<typename Basis, std::size_t n>
void NitscheCondition<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& values,
    const ElementValues<Basis, n>& shifted,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    Eigen::MatrixX<Scalar> trace;
    Eigen::MatrixX<Scalar> shifted_trace;
    Eigen::MatrixX<Scalar> flux;
    trace_.local_trace(values, trace);
    trace_.local_trace(shifted, shifted_trace);
    flux_.local_flux(values, normals, flux);

    const Eigen::VectorX<Scalar> penalized =
        penalty_ * weights.cwiseQuotient(values.sizes());

    // K_ij = − ∫ F(N_j) T(N_i) ds − ∫ F(N_i) T(S N_j) ds
    //        + γ ∫ T(S N_i) T(S N_j) / h ds
    //      ≈ Σ_q w_q (γ B̃_iq B̃_jq / h_q − B_iq F_jq − F_iq B̃_jq)
    stiffness =
        shifted_trace * penalized.asDiagonal() * shifted_trace.transpose()
        - trace * weights.asDiagonal() * flux.transpose()
        - flux * weights.asDiagonal() * shifted_trace.transpose();
}

template<typename Basis, std::size_t n>
void NitscheCondition<Basis, n>::local_load(
    const ElementValues<Basis, n>& values,
    const ElementValues<Basis, n>& shifted,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals,
    const Eigen::VectorX<Scalar>& data,
    Eigen::VectorX<Scalar>& load) const
{
    Eigen::MatrixX<Scalar> shifted_trace;
    Eigen::MatrixX<Scalar> flux;
    trace_.local_trace(shifted, shifted_trace);
    flux_.local_flux(values, normals, flux);

    const Eigen::VectorX<Scalar> penalized =
        penalty_ * weights.cwiseQuotient(values.sizes());

    // F_i = ∫ (γ T(S N_i) / h − F(N_i)) g ds
    //     ≈ Σ_q w_q g(x_q) (γ B̃_iq / h_q − F_iq)
    load = shifted_trace * penalized.cwiseProduct(data)
           - flux * weights.cwiseProduct(data);
}

template class NitscheCondition<TensorBSpline<double, 2>, 2>;
template class NitscheCondition<TensorBSpline<double, 2>, 3>;
template class NitscheCondition<TensorBSpline<double, 3>, 3>;
template class NitscheCondition<TensorNURBS<double, 2>, 2>;
template class NitscheCondition<TensorNURBS<double, 2>, 3>;
template class NitscheCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
