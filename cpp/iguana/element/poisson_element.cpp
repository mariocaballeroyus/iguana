/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "poisson_element.hpp"

#include <cstddef>

namespace iguana
{

template<typename Basis, std::size_t n>
void PoissonElement<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    const Eigen::Index num_active = values.values().rows();
    stiffness.setZero(num_active, num_active);

    // K_ij = ∫ ∇N_i · ∇N_j dx
    //      ≈ Σ_q w_q ∇N_i(x_q) · ∇N_j(x_q)
    for (const Eigen::MatrixX<Scalar>& derivs : values.physical_gradients())
        stiffness += derivs * weights.asDiagonal() * derivs.transpose();
}

template<typename Basis, std::size_t n>
void PoissonElement<Basis, n>::local_load(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::VectorX<Scalar>& source,
    Eigen::VectorX<Scalar>& load) const
{
    // F_i = ∫ f N_i dx
    //     ≈ Σ_q w_q f(x_q) N_i(x_q)
    load = values.values() * weights.cwiseProduct(source);
}

template class PoissonElement<TensorBSpline<double, 2>, 2>;
template class PoissonElement<TensorBSpline<double, 3>, 3>;
template class PoissonElement<TensorNURBS<double, 2>, 2>;
template class PoissonElement<TensorNURBS<double, 3>, 3>;

} // namespace iguana
