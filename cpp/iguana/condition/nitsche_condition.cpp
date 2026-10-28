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
                                             Scalar penalty, bool symmetric)
    : trace_(trace), flux_(flux), penalty_(penalty), symmetric_(symmetric)
{
    // Only the symmetric method needs a penalty to be stable
    if (symmetric && penalty <= 0)
        throw std::invalid_argument("NitscheCondition: "
                                    "the penalty must be positive");

    if (penalty < 0)
        throw std::invalid_argument("NitscheCondition: "
                                    "the penalty must be non-negative");
}

template<typename Basis, std::size_t n>
void NitscheCondition<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const ElementValues<Basis, n>& shifted,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals, const Eigen::VectorX<Scalar>&,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    // T(N_i), traces of the test functions
    Eigen::MatrixX<Scalar> trace;
    trace_.local_trace(values, trace);

    // T(S N_i), traces where the data is given
    Eigen::MatrixX<Scalar> shifted_trace;
    trace_.local_trace(shifted, shifted_trace);

    // F(N_i), fluxes along the normal
    Eigen::MatrixX<Scalar> flux;
    flux_.local_flux(values, normals, flux);

    // Penalty over the size h of the element
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * weights.cwiseQuotient(values.sizes());

    // The sign s of the adjoint term, −1 in the symmetric method and +1 in
    // the skew-symmetric one
    const Scalar adjoint = symmetric_ ? -1 : 1;

    // K_ij = − ( T(N_i), F(N_j) )_Γ
    //        + s ( F(N_i), T(S N_j) )_Γ
    //        + β ( T(S N_i), T(S N_j) / h )_Γ
    //      ≈ Σ_q w_q (β B̃_iq B̃_jq / h_q − B_iq F_jq + s F_iq B̃_jq)
    tangent =
        shifted_trace * penalized.asDiagonal() * shifted_trace.transpose()
        - trace * weights.asDiagonal() * flux.transpose()
        + adjoint * flux * weights.asDiagonal() * shifted_trace.transpose();
}

template<typename Basis, std::size_t n>
void NitscheCondition<Basis, n>::local_residual(
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
    Eigen::MatrixX<Scalar> shifted_trace;
    trace_.local_trace(shifted, shifted_trace);

    // F(N_i), fluxes along the normal
    Eigen::MatrixX<Scalar> flux;
    flux_.local_flux(values, normals, flux);

    // Penalty over the size h of the element
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * weights.cwiseQuotient(values.sizes());

    // The sign s of the adjoint term, −1 in the symmetric method and +1 in
    // the skew-symmetric one
    const Scalar adjoint = symmetric_ ? -1 : 1;

    // F_i = β ( T(S N_i) / h, g )_Γ + s ( F(N_i), g )_Γ
    //     ≈ Σ_q w_q g(x_q) (β B̃_iq / h_q + s F_iq)
    const Eigen::VectorX<Scalar> load =
        shifted_trace * penalized.cwiseProduct(data)
        + adjoint * flux * weights.cwiseProduct(data);

    // R_i = Σ_j K_ij c_j − F_i
    residual = tangent * coefficients - load;
}

template class NitscheCondition<TensorBSpline<double, 2>, 2>;
template class NitscheCondition<TensorBSpline<double, 2>, 3>;
template class NitscheCondition<TensorBSpline<double, 3>, 3>;
template class NitscheCondition<TensorNURBS<double, 2>, 2>;
template class NitscheCondition<TensorNURBS<double, 2>, 3>;
template class NitscheCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
