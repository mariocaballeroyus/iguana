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
    : NitscheCondition(trace, flux, trace, flux, penalty, symmetric)
{
}

template<typename Basis, std::size_t n>
NitscheCondition<Basis, n>::NitscheCondition(
    const Trace<Basis, n>& trace, const Flux<Basis, n>& flux,
    const Trace<Basis, n>& test_trace, const Flux<Basis, n>& test_flux,
    Scalar penalty, bool symmetric)
    : trace_(trace), flux_(flux), test_trace_(test_trace),
      test_flux_(test_flux), penalty_(penalty), symmetric_(symmetric)
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
    // Tᵗ(N_i), traces of the test functions
    Eigen::MatrixX<Scalar> trace;
    test_trace_.local_trace(values, trace);

    // T(S N_j) and Tᵗ(S N_i), traces where the data is given
    Eigen::MatrixX<Scalar> shifted_trace;
    trace_.local_trace(shifted, shifted_trace);

    Eigen::MatrixX<Scalar> test_shifted;
    test_trace_.local_trace(shifted, test_shifted);

    // F(N_j) and Fᵗ(N_i), fluxes along the normal
    Eigen::MatrixX<Scalar> flux;
    flux_.local_flux(values, normals, flux);

    Eigen::MatrixX<Scalar> test_flux;
    test_flux_.local_flux(values, normals, test_flux);

    // Penalty over the size h of the element
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * weights.cwiseQuotient(values.sizes());

    // The sign s of the adjoint term, −1 in the symmetric method and +1 in
    // the skew-symmetric one
    const Scalar adjoint = symmetric_ ? -1 : 1;

    // K_ij = − ( Tᵗ(N_i), F(N_j) )_Γ
    //        + s ( Fᵗ(N_i), T(S N_j) )_Γ
    //        + β ( Tᵗ(S N_i), T(S N_j) / h )_Γ
    //      ≈ Σ_q w_q (β B̃ᵗ_iq B̃_jq / h_q − Bᵗ_iq F_jq + s Fᵗ_iq B̃_jq)
    tangent =
        test_shifted * penalized.asDiagonal() * shifted_trace.transpose()
        - trace * weights.asDiagonal() * flux.transpose()
        + adjoint * test_flux * weights.asDiagonal()
              * shifted_trace.transpose();
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

    // Tᵗ(S N_i), traces of the test functions where the data is given
    Eigen::MatrixX<Scalar> test_shifted;
    test_trace_.local_trace(shifted, test_shifted);

    // Fᵗ(N_i), fluxes of the test functions along the normal
    Eigen::MatrixX<Scalar> test_flux;
    test_flux_.local_flux(values, normals, test_flux);

    // Penalty over the size h of the element
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * weights.cwiseQuotient(values.sizes());

    // The sign s of the adjoint term, −1 in the symmetric method and +1 in
    // the skew-symmetric one
    const Scalar adjoint = symmetric_ ? -1 : 1;

    // F_i = β ( Tᵗ(S N_i) / h, g )_Γ + s ( Fᵗ(N_i), g )_Γ
    //     ≈ Σ_q w_q g(x_q) (β B̃ᵗ_iq / h_q + s Fᵗ_iq)
    const Eigen::VectorX<Scalar> load =
        test_shifted * penalized.cwiseProduct(data)
        + adjoint * test_flux * weights.cwiseProduct(data);

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
