/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "stokes_element.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
StokesElement<Basis, n>::StokesElement(Scalar viscosity,
                                       Scalar stabilization)
    : viscosity_(viscosity), stabilization_(stabilization)
{
    if (viscosity <= 0)
        throw std::invalid_argument("StokesElement: "
                                    "the viscosity must be positive");

    if (stabilization <= 0)
        throw std::invalid_argument("StokesElement: "
                                    "the stabilization must be positive");
}

template<typename Basis, std::size_t n>
void StokesElement<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    const std::array<Eigen::MatrixX<Scalar>, n>& grads =
        values.physical_gradients();
    const std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>& hessians =
        values.physical_hessians();
    const Eigen::Index num_active = values.values().rows();
    const Eigen::Index pressure = static_cast<Eigen::Index>(n) * num_active;

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    // ∇N_i · ∇N_j and τ_M ∇N_i · ∇N_j, and ΔN_j
    Eigen::MatrixX<Scalar> laplacian =
        Eigen::MatrixX<Scalar>::Zero(num_active, num_active);
    Eigen::MatrixX<Scalar> stabilized_laplacian = laplacian;
    Eigen::MatrixX<Scalar> laplacians =
        Eigen::MatrixX<Scalar>::Zero(num_active, values.values().cols());

    for (std::size_t coordinate = 0; coordinate < n; ++coordinate) {
        laplacian += grads[coordinate] * weights.asDiagonal()
                     * grads[coordinate].transpose();
        stabilized_laplacian += grads[coordinate] * stabilized.asDiagonal()
                                * grads[coordinate].transpose();
        laplacians += hessians[coordinate];
    }

    stiffness.setZero(pressure + num_active, pressure + num_active);

    // Block (a, b) pairs component a of the test functions with component b
    // of the trial ones, the pressure following the velocity
    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(a) * num_active;

        for (std::size_t b = 0; b < n; ++b) {
            const Eigen::Index col = static_cast<Eigen::Index>(b) * num_active;

            // K_ij = ( 2μ ε(N_i e_a), ε(N_j e_b) )_Ω
            //        + ( τ_C div(N_i e_a), div(N_j e_b) )_Ω
            //      ≈ Σ_q w_q μ (δ_ab ∇N_i · ∇N_j + ∂_b N_i ∂_a N_j
            //                   + ∂_a N_i ∂_b N_j)
            stiffness.block(row, col, num_active, num_active) =
                viscosity_ * (grads[b] * weights.asDiagonal()
                                  * grads[a].transpose()
                              + grads[a] * weights.asDiagonal()
                                    * grads[b].transpose());

            if (a == b)
                stiffness.block(row, col, num_active, num_active) +=
                    viscosity_ * laplacian;
        }

        // K_ij = − ( div(N_i e_a), N_j )_Ω
        //      ≈ − Σ_q w_q ∂_a N_i N_j
        stiffness.block(row, pressure, num_active, num_active) =
            -grads[a] * weights.asDiagonal() * values.values().transpose();
    }

    for (std::size_t b = 0; b < n; ++b) {
        const Eigen::Index col = static_cast<Eigen::Index>(b) * num_active;

        // ∇N_i · ∇∂_b N_j, from the gradient of the divergence
        Eigen::MatrixX<Scalar> curvature =
            Eigen::MatrixX<Scalar>::Zero(num_active, num_active);

        for (std::size_t a = 0; a < n; ++a)
            curvature += grads[a] * stabilized.asDiagonal()
                         * hessians[pair_index<n>(a, b)].transpose();

        // K_ij = − ( N_i, div(N_j e_b) )_Ω
        //        − ( τ_M ∇N_i, −div(2μ ε(N_j e_b)) )_Ω
        //      ≈ Σ_q w_q (τ_M μ (∂_b N_i ΔN_j + ∇N_i · ∇∂_b N_j)
        //                 − N_i ∂_b N_j)
        stiffness.block(pressure, col, num_active, num_active) =
            viscosity_ * (grads[b] * stabilized.asDiagonal()
                              * laplacians.transpose()
                          + curvature)
            - values.values() * weights.asDiagonal() * grads[b].transpose();
    }

    // K_ij = − ( τ_M ∇N_i, ∇N_j )_Ω
    //      ≈ − Σ_q w_q τ_M ∇N_i · ∇N_j
    stiffness.block(pressure, pressure, num_active, num_active) =
        -stabilized_laplacian;
}

template<typename Basis, std::size_t n>
void StokesElement<Basis, n>::local_load(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    Eigen::VectorX<Scalar>& load) const
{
    const Eigen::Index num_active = values.values().rows();
    const Eigen::Index pressure = static_cast<Eigen::Index>(n) * num_active;

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    load.setZero(pressure + num_active);

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(a) * num_active;
        const Eigen::Index component = static_cast<Eigen::Index>(a);

        // F_i = ( N_i, f_a )_Ω
        //     ≈ Σ_q w_q f_a(x_q) N_i(x_q)
        load.segment(row, num_active) =
            values.values() * weights.cwiseProduct(source.col(component));

        // F_i = − ( τ_M ∇N_i, f )_Ω
        //     ≈ − Σ_q w_q τ_M ∇N_i(x_q) · f(x_q)
        load.segment(pressure, num_active) -=
            values.physical_gradients()[a]
            * stabilized.cwiseProduct(source.col(component));
    }
}

template<typename Basis, std::size_t n>
void StokesElement<Basis, n>::local_mass(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& mass) const
{
    const Eigen::Index num_active = values.values().rows();
    const Eigen::Index pressure = static_cast<Eigen::Index>(n) * num_active;

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    mass.setZero(pressure + num_active, pressure + num_active);

    for (std::size_t b = 0; b < n; ++b) {
        const Eigen::Index col = static_cast<Eigen::Index>(b) * num_active;

        // M_ij = ( N_i e_b, N_j e_b )_Ω
        //      ≈ Σ_q w_q N_i(x_q) N_j(x_q)
        mass.block(col, col, num_active, num_active) =
            values.values() * weights.asDiagonal()
            * values.values().transpose();

        // M_ij = − ( τ_M ∇N_i, N_j e_b )_Ω
        //      ≈ − Σ_q w_q τ_M ∂_b N_i(x_q) N_j(x_q)
        mass.block(pressure, col, num_active, num_active) =
            -values.physical_gradients()[b] * stabilized.asDiagonal()
            * values.values().transpose();
    }
}

template<typename Basis, std::size_t n>
Eigen::VectorX<typename Basis::Scalar> StokesElement<Basis, n>::tau(
    const ElementValues<Basis, n>& values) const
{
    // τ_M = h² / (c μ)
    return values.sizes().array().square() / (stabilization_ * viscosity_);
}

template class StokesElement<TensorBSpline<double, 2>, 2>;
template class StokesElement<TensorBSpline<double, 3>, 3>;
template class StokesElement<TensorNURBS<double, 2>, 2>;
template class StokesElement<TensorNURBS<double, 3>, 3>;

} // namespace iguana
