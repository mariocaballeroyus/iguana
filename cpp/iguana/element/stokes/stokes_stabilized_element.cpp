/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "stokes_stabilized_element.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
StokesStabilizedElement<Basis, n>::StokesStabilizedElement(
    Scalar viscosity, Scalar stabilization)
    : Base(viscosity), stabilization_(stabilization)
{
    if (stabilization <= 0)
        throw std::invalid_argument("StokesStabilizedElement: "
                                    "the stabilization must be positive");
}

template<typename Basis, std::size_t n>
void StokesStabilizedElement<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    Base::local_stiffness(values, weights, stiffness);

    const std::array<Eigen::MatrixX<Scalar>, n>& grads =
        values.physical_gradients();
    const std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>& hessians =
        values.physical_hessians();
    const Eigen::Index num_active = values.values().rows();
    const Eigen::Index pressure = static_cast<Eigen::Index>(n) * num_active;
    const Scalar viscosity = this->viscosity();

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    // τ_M ∇N_i · ∇N_j, and ΔN_j
    Eigen::MatrixX<Scalar> stabilized_laplacian =
        Eigen::MatrixX<Scalar>::Zero(num_active, num_active);
    Eigen::MatrixX<Scalar> laplacians =
        Eigen::MatrixX<Scalar>::Zero(num_active, values.values().cols());

    for (std::size_t coordinate = 0; coordinate < n; ++coordinate) {
        stabilized_laplacian += grads[coordinate] * stabilized.asDiagonal()
                                * grads[coordinate].transpose();
        laplacians += hessians[coordinate];
    }

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(a) * num_active;

        for (std::size_t b = 0; b < n; ++b) {
            const Eigen::Index col = static_cast<Eigen::Index>(b) * num_active;

            // K_ij = ( τ_C div(N_i e_a), div(N_j e_b) )_Ω
            //      ≈ Σ_q w_q μ ∂_a N_i ∂_b N_j
            stiffness.block(row, col, num_active, num_active) +=
                viscosity * grads[a] * weights.asDiagonal()
                * grads[b].transpose();
        }
    }

    for (std::size_t b = 0; b < n; ++b) {
        const Eigen::Index col = static_cast<Eigen::Index>(b) * num_active;

        // ∇N_i · ∇∂_b N_j, from the gradient of the divergence
        Eigen::MatrixX<Scalar> curvature =
            Eigen::MatrixX<Scalar>::Zero(num_active, num_active);

        for (std::size_t a = 0; a < n; ++a)
            curvature += grads[a] * stabilized.asDiagonal()
                         * hessians[pair_index<n>(a, b)].transpose();

        // K_ij = − ( τ_M ∇N_i, −div(2μ ε(N_j e_b)) )_Ω
        //      ≈ Σ_q w_q τ_M μ (∂_b N_i ΔN_j + ∇N_i · ∇∂_b N_j)
        stiffness.block(pressure, col, num_active, num_active) +=
            viscosity * (grads[b] * stabilized.asDiagonal()
                             * laplacians.transpose()
                         + curvature);
    }

    // K_ij = − ( τ_M ∇N_i, ∇N_j )_Ω
    //      ≈ − Σ_q w_q τ_M ∇N_i · ∇N_j
    stiffness.block(pressure, pressure, num_active, num_active) -=
        stabilized_laplacian;
}

template<typename Basis, std::size_t n>
void StokesStabilizedElement<Basis, n>::local_load(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    Eigen::VectorX<Scalar>& load) const
{
    Base::local_load(values, weights, source, load);

    const Eigen::Index num_active = values.values().rows();
    const Eigen::Index pressure = static_cast<Eigen::Index>(n) * num_active;

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index component = static_cast<Eigen::Index>(a);

        // F_i = − ( τ_M ∇N_i, f )_Ω
        //     ≈ − Σ_q w_q τ_M ∇N_i(x_q) · f(x_q)
        load.segment(pressure, num_active) -=
            values.physical_gradients()[a]
            * stabilized.cwiseProduct(source.col(component));
    }
}

template<typename Basis, std::size_t n>
void StokesStabilizedElement<Basis, n>::local_mass(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& mass) const
{
    Base::local_mass(values, weights, mass);

    const Eigen::Index num_active = values.values().rows();
    const Eigen::Index pressure = static_cast<Eigen::Index>(n) * num_active;

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    for (std::size_t b = 0; b < n; ++b) {
        const Eigen::Index col = static_cast<Eigen::Index>(b) * num_active;

        // M_ij = − ( τ_M ∇N_i, N_j e_b )_Ω
        //      ≈ − Σ_q w_q τ_M ∂_b N_i(x_q) N_j(x_q)
        mass.block(pressure, col, num_active, num_active) -=
            values.physical_gradients()[b] * stabilized.asDiagonal()
            * values.values().transpose();
    }
}

template<typename Basis, std::size_t n>
Eigen::VectorX<typename Basis::Scalar> StokesStabilizedElement<Basis, n>::tau(
    const ElementValues<Basis, n>& values) const
{
    // τ_M = h² / (c μ)
    return values.sizes().array().square()
           / (stabilization_ * this->viscosity());
}

template class StokesStabilizedElement<TensorBSpline<double, 2>, 2>;
template class StokesStabilizedElement<TensorBSpline<double, 3>, 3>;
template class StokesStabilizedElement<TensorNURBS<double, 2>, 2>;
template class StokesStabilizedElement<TensorNURBS<double, 3>, 3>;

} // namespace iguana
