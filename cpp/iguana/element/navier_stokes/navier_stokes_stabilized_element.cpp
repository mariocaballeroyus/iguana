/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "navier_stokes_stabilized_element.hpp"

#include <array>
#include <cstddef>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
NavierStokesStabilizedElement<Basis, n>::NavierStokesStabilizedElement(
    Scalar viscosity, Scalar stabilization)
    : Base(viscosity, stabilization)
{
}

template<typename Basis, std::size_t n>
void NavierStokesStabilizedElement<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::MatrixX<Scalar>& tangent) const
{
    Base::local_tangent(values, weights, source, coefficients, tangent);

    const std::size_t num_fields = values.num_fields();
    const typename Base::Flow state =
        this->flow(values, source, coefficients);
    const Eigen::ArrayX<Scalar> w = weights.array();
    const Eigen::ArrayX<Scalar> h = values.sizes().array();
    const Scalar c = this->stabilization();

    const std::array<Eigen::MatrixX<Scalar>, n>& pressure_grads =
        values.physical_gradients(Base::field_of(num_fields, n));
    const Eigen::Index pressure_row = Base::first_row(values, n);
    const Eigen::Index pressure_rows = pressure_grads[0].rows();

    // τ_C = h² / (c τ_M), and div u
    const Eigen::ArrayX<Scalar> tau_c = h.square() / (c * state.tau);
    Eigen::ArrayX<Scalar> divergence = Eigen::ArrayX<Scalar>::Zero(w.size());

    for (std::size_t k = 0; k < n; ++k)
        divergence += state.gradient[k][k];

    // Physical weights times the derivatives of τ_M and τ_C with |u| over
    // |u|: δτ_M = − τ_M² (2 / h) δ|u| and δτ_C = (2 h / c) δ|u|, with
    // δ|u| = u_b N_j / |u|
    const Eigen::ArrayX<Scalar> tau_change =
        -2 * w * state.tau.square() / (h * state.speed);
    const Eigen::ArrayX<Scalar> tau_c_change = 2 * w * h / (c * state.speed);

    for (std::size_t b = 0; b < n; ++b) {
        const std::size_t trial_field = Base::field_of(num_fields, b);
        const Eigen::MatrixX<Scalar>& trial_functions =
            values.values(trial_field);
        const std::array<Eigen::MatrixX<Scalar>, n>& trial_grads =
            values.physical_gradients(trial_field);
        const Eigen::Index col = Base::first_row(values, b);
        const Eigen::Index cols = trial_functions.rows();

        for (std::size_t a = 0; a < n; ++a) {
            const std::array<Eigen::MatrixX<Scalar>, n>& test_grads =
                values.physical_gradients(Base::field_of(num_fields, a));
            const Eigen::Index row = Base::first_row(values, a);
            const Eigen::Index rows = test_grads[a].rows();

            // K_ij = ( τ_C div(N_i e_a), div(N_j e_b) )_Ω
            //      ≈ Σ_q w_q τ_C ∂_a N_i ∂_b N_j
            tangent.block(row, col, rows, cols) +=
                test_grads[a] * (w * tau_c).matrix().asDiagonal()
                * trial_grads[b].transpose();

            // K_ij = ( δτ_C div u, div(N_i e_a) )_Ω
            //      ≈ Σ_q w_q (2 h / (c |u|)) u_b N_j div u ∂_a N_i
            tangent.block(row, col, rows, cols) +=
                test_grads[a]
                * (tau_c_change * state.velocity[b] * divergence).matrix()
                      .asDiagonal()
                * trial_functions.transpose();

            // K_ij = − ( τ_M ∇N_i, ∂r_M / ∂c_j )_Ω
            //      ≈ − Σ_q w_q τ_M ∂_a N_i ∂r_a / ∂c_j
            tangent.block(pressure_row, col, pressure_rows, cols) -=
                pressure_grads[a] * (w * state.tau).matrix().asDiagonal()
                * this->momentum_derivative(values, state, a, b).transpose();

            // K_ij = − ( δτ_M ∇N_i, r_M )_Ω
            //      ≈ Σ_q w_q τ_M² (2 / (h |u|)) u_b N_j ∂_a N_i r_a
            tangent.block(pressure_row, col, pressure_rows, cols) -=
                pressure_grads[a]
                * (tau_change * state.velocity[b] * state.momentum[a])
                      .matrix().asDiagonal()
                * trial_functions.transpose();
        }
    }

    // K_ij = − ( τ_M ∇N_i, ∇N_j )_Ω
    //      ≈ − Σ_q w_q τ_M ∇N_i · ∇N_j
    for (std::size_t k = 0; k < n; ++k) {
        tangent.block(pressure_row, pressure_row, pressure_rows,
                      pressure_rows) -=
            pressure_grads[k] * (w * state.tau).matrix().asDiagonal()
            * pressure_grads[k].transpose();
    }
}

template<typename Basis, std::size_t n>
void NavierStokesStabilizedElement<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Base::local_residual(values, weights, source, coefficients, residual);

    const std::size_t num_fields = values.num_fields();
    const typename Base::Flow state =
        this->flow(values, source, coefficients);
    const Eigen::ArrayX<Scalar> w = weights.array();
    const Eigen::ArrayX<Scalar> h = values.sizes().array();

    const std::array<Eigen::MatrixX<Scalar>, n>& pressure_grads =
        values.physical_gradients(Base::field_of(num_fields, n));
    const Eigen::Index pressure_row = Base::first_row(values, n);

    // τ_C = h² / (c τ_M), and div u
    const Eigen::ArrayX<Scalar> tau_c =
        h.square() / (this->stabilization() * state.tau);
    Eigen::ArrayX<Scalar> divergence = Eigen::ArrayX<Scalar>::Zero(w.size());

    for (std::size_t k = 0; k < n; ++k)
        divergence += state.gradient[k][k];

    for (std::size_t a = 0; a < n; ++a) {
        const std::array<Eigen::MatrixX<Scalar>, n>& grads =
            values.physical_gradients(Base::field_of(num_fields, a));

        // R_i = ( τ_C div u, div(N_i e_a) )_Ω
        //     ≈ Σ_q w_q τ_C div u ∂_a N_i
        residual.segment(Base::first_row(values, a), grads[a].rows()) +=
            grads[a] * (w * tau_c * divergence).matrix();

        // R_i = − ( τ_M ∇N_i, r_M )_Ω
        //     ≈ − Σ_q w_q τ_M ∂_a N_i r_a
        residual.segment(pressure_row, pressure_grads[a].rows()) -=
            pressure_grads[a] * (w * state.tau * state.momentum[a]).matrix();
    }
}

template class NavierStokesStabilizedElement<TensorBSpline<double, 2>, 2>;
template class NavierStokesStabilizedElement<TensorBSpline<double, 3>, 3>;
template class NavierStokesStabilizedElement<TensorNURBS<double, 2>, 2>;
template class NavierStokesStabilizedElement<TensorNURBS<double, 3>, 3>;
template class NavierStokesStabilizedElement<THBSpline<double, 2>, 2>;

} // namespace iguana
