/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "poisson_element.hpp"

#include <cstddef>

namespace iguana
{

template<typename Basis, std::size_t n>
void PoissonElement<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights, const Eigen::MatrixX<Scalar>&,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    const Eigen::Index num_active = values.values().rows();
    tangent.setZero(num_active, num_active);

    // K_ij = ( ∇N_i, ∇N_j )_Ω
    //      ≈ Σ_q w_q ∇N_i(x_q) · ∇N_j(x_q)
    for (const Eigen::MatrixX<Scalar>& derivs : values.physical_gradients())
        tangent += derivs * weights.asDiagonal() * derivs.transpose();
}

template<typename Basis, std::size_t n>
void PoissonElement<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Eigen::MatrixX<Scalar> tangent;
    local_tangent(values, weights, source, coefficients, tangent);

    // R_i = ( ∇N_i, ∇u_h )_Ω − ( N_i, f )_Ω
    //     ≈ Σ_j K_ij c_j − Σ_q w_q f(x_q) N_i(x_q)
    residual = tangent * coefficients
               - values.values() * weights.cwiseProduct(source.col(0));
}

template<typename Basis, std::size_t n>
void PoissonElement<Basis, n>::local_mass(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& mass) const
{
    // M_ij = ( N_i, N_j )_Ω
    //      ≈ Σ_q w_q N_i(x_q) N_j(x_q)
    mass = values.values() * weights.asDiagonal()
           * values.values().transpose();
}

template class PoissonElement<TensorBSpline<double, 2>, 2>;
template class PoissonElement<TensorBSpline<double, 3>, 3>;
template class PoissonElement<TensorNURBS<double, 2>, 2>;
template class PoissonElement<TensorNURBS<double, 3>, 3>;

} // namespace iguana
