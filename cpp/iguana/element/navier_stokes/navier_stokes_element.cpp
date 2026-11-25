/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "navier_stokes_element.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
NavierStokesElement<Basis, n>::NavierStokesElement(Scalar viscosity,
                                                   Scalar stabilization)
    : Base(viscosity), stabilization_(stabilization)
{
    if (stabilization <= 0)
        throw std::invalid_argument("NavierStokesElement: "
                                    "the stabilization must be positive");
}

template<typename Basis, std::size_t n>
void NavierStokesElement<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::MatrixX<Scalar>& tangent) const
{
    Base::local_tangent(values, weights, source, coefficients, tangent);

    const std::size_t num_fields = values.num_fields();
    const Flow state = flow(values, source, coefficients);
    const Eigen::ArrayX<Scalar> w = weights.array();
    const Eigen::ArrayX<Scalar> h = values.sizes().array();

    const std::array<Eigen::MatrixX<Scalar>, n>& pressure_grads =
        values.physical_gradients(Base::field_of(num_fields, n));
    const Eigen::Index pressure_row = Base::first_row(values, n);
    const Eigen::Index pressure_rows = pressure_grads[0].rows();

    // Physical weights times ∂τ_M / ∂|u| / |u|: δτ_M = − τ_M² (2 / h) δ|u|,
    // with δ|u| = u_b N_j / |u|
    const Eigen::ArrayX<Scalar> tau_change =
        -2 * w * state.tau.square() / (h * state.speed);

    for (std::size_t a = 0; a < n; ++a) {
        const std::size_t test_field = Base::field_of(num_fields, a);
        const Eigen::MatrixX<Scalar>& test_functions =
            values.values(test_field);
        const std::array<Eigen::MatrixX<Scalar>, n>& test_grads =
            values.physical_gradients(test_field);
        const Eigen::MatrixX<Scalar> test_convected =
            convected(test_grads, state);
        const Eigen::Index row = Base::first_row(values, a);
        const Eigen::Index rows = test_functions.rows();

        for (std::size_t b = 0; b < n; ++b) {
            const std::size_t trial_field = Base::field_of(num_fields, b);
            const Eigen::MatrixX<Scalar>& trial_functions =
                values.values(trial_field);
            auto block = tangent.block(row, Base::first_row(values, b), rows,
                                       trial_functions.rows());

            // K_ij = ( (u · ∇)(N_j e_b) + (N_j e_b · ∇)u, N_i e_a )_Ω
            //      ≈ Σ_q w_q (δ_ab u · ∇N_j + N_j ∂_b u_a) N_i
            block += test_functions
                     * (w * state.gradient[a][b]).matrix().asDiagonal()
                     * trial_functions.transpose();

            if (a == b) {
                block += test_functions * w.matrix().asDiagonal()
                         * convected(values.physical_gradients(trial_field),
                                     state)
                               .transpose();
            }

            // K_ij = ( τ_M (u · ∇)N_i, ∂r_a / ∂c_j )_Ω
            //      ≈ Σ_q w_q τ_M (u · ∇N_i) ∂r_a / ∂c_j
            block += test_convected * (w * state.tau).matrix().asDiagonal()
                     * momentum_derivative(values, state, a, b).transpose();

            // K_ij = ( τ_M (N_j e_b · ∇)N_i, r_a )_Ω
            //      ≈ Σ_q w_q τ_M N_j ∂_b N_i r_a
            block += test_grads[b]
                     * (w * state.tau * state.momentum[a]).matrix()
                           .asDiagonal()
                     * trial_functions.transpose();

            // K_ij = ( δτ_M (u · ∇)N_i, r_a )_Ω
            //      ≈ − Σ_q w_q τ_M² (2 / (h |u|)) u_b N_j (u · ∇N_i) r_a
            block += test_convected
                     * (tau_change * state.velocity[b] * state.momentum[a])
                           .matrix().asDiagonal()
                     * trial_functions.transpose();
        }

        // K_ij = ( τ_M (u · ∇)N_i, ∂_a N_j )_Ω
        //      ≈ Σ_q w_q τ_M (u · ∇N_i) ∂_a N_j
        tangent.block(row, pressure_row, rows, pressure_rows) +=
            test_convected * (w * state.tau).matrix().asDiagonal()
            * pressure_grads[a].transpose();
    }
}

template<typename Basis, std::size_t n>
void NavierStokesElement<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Base::local_residual(values, weights, source, coefficients, residual);

    const Flow state = flow(values, source, coefficients);
    const Eigen::ArrayX<Scalar> w = weights.array();

    for (std::size_t a = 0; a < n; ++a) {
        const std::size_t field = Base::field_of(values.num_fields(), a);
        const Eigen::MatrixX<Scalar>& functions = values.values(field);
        const std::array<Eigen::MatrixX<Scalar>, n>& grads =
            values.physical_gradients(field);
        auto rows = residual.segment(Base::first_row(values, a),
                                     functions.rows());

        Eigen::ArrayX<Scalar> convection =
            Eigen::ArrayX<Scalar>::Zero(w.size());

        for (std::size_t k = 0; k < n; ++k)
            convection += state.velocity[k] * state.gradient[a][k];

        // R_i = ( (u · ∇)u_a, N_i )_Ω
        //     ≈ Σ_q w_q (u · ∇u_a) N_i
        rows += functions * (w * convection).matrix();

        // R_i = ( τ_M (u · ∇)N_i, r_a )_Ω
        //     ≈ Σ_q w_q τ_M (u · ∇N_i) r_a
        for (std::size_t k = 0; k < n; ++k)
            rows += grads[k]
                    * (w * state.tau * state.velocity[k] * state.momentum[a])
                          .matrix();
    }
}

template<typename Basis, std::size_t n>
typename NavierStokesElement<Basis, n>::Flow
NavierStokesElement<Basis, n>::flow(
    const ElementValues<Basis, n>& values,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients) const
{
    const std::size_t num_fields = values.num_fields();
    const Eigen::ArrayX<Scalar> h = values.sizes().array();
    const Scalar viscosity = this->viscosity();
    Flow result;

    // The coefficients of each component of the velocity, then the velocity
    // and its gradient
    std::array<Eigen::VectorX<Scalar>, n> components;

    for (std::size_t a = 0; a < n; ++a) {
        const std::size_t field = Base::field_of(num_fields, a);
        const std::array<Eigen::MatrixX<Scalar>, n>& grads =
            values.physical_gradients(field);

        components[a] = coefficients.segment(Base::first_row(values, a),
                                             values.values(field).rows());
        result.velocity[a] =
            (values.values(field).transpose() * components[a]).array();

        for (std::size_t k = 0; k < n; ++k)
            result.gradient[a][k] =
                (grads[k].transpose() * components[a]).array();
    }

    const std::size_t pressure_field = Base::field_of(num_fields, n);
    const std::array<Eigen::MatrixX<Scalar>, n>& pressure_grads =
        values.physical_gradients(pressure_field);
    const Eigen::VectorX<Scalar> pressure =
        coefficients.segment(Base::first_row(values, n),
                             values.values(pressure_field).rows());

    // r_a = (u · ∇)u_a − μ (Δu_a + ∂_a div u) + ∂_a p − f_a
    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index component = static_cast<Eigen::Index>(a);

        result.momentum[a] = (pressure_grads[a].transpose() * pressure).array()
                             - source.col(component).array();

        for (std::size_t k = 0; k < n; ++k) {
            const std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>&
                own = values.physical_hessians(Base::field_of(num_fields, a));
            const std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>&
                other =
                    values.physical_hessians(Base::field_of(num_fields, k));

            result.momentum[a] +=
                result.velocity[k] * result.gradient[a][k]
                - viscosity
                      * ((own[pair_index<n>(k, k)].transpose()
                          * components[a])
                         + other[pair_index<n>(a, k)].transpose()
                               * components[k])
                            .array();
        }
    }

    // |u| = sqrt(u · u + ε²), and τ_M = 1 / (c μ / h² + 2 |u| / h)
    const Scalar floor = std::numeric_limits<Scalar>::epsilon();
    Eigen::ArrayX<Scalar> squared =
        Eigen::ArrayX<Scalar>::Constant(h.size(), floor * floor);

    for (std::size_t a = 0; a < n; ++a)
        squared += result.velocity[a].square();

    result.speed = squared.sqrt();
    result.tau = 1 / (stabilization_ * viscosity / h.square()
                      + 2 * result.speed / h);

    return result;
}

template<typename Basis, std::size_t n>
Eigen::MatrixX<typename Basis::Scalar> NavierStokesElement<Basis, n>::convected(
    const std::array<Eigen::MatrixX<Scalar>, n>& grads, const Flow& state)
{
    Eigen::MatrixX<Scalar> result =
        Eigen::MatrixX<Scalar>::Zero(grads[0].rows(), grads[0].cols());

    for (std::size_t k = 0; k < n; ++k)
        result += grads[k] * state.velocity[k].matrix().asDiagonal();

    return result;
}

template<typename Basis, std::size_t n>
Eigen::MatrixX<typename Basis::Scalar>
NavierStokesElement<Basis, n>::momentum_derivative(
    const ElementValues<Basis, n>& values, const Flow& state, std::size_t a,
    std::size_t b) const
{
    const std::size_t field = Base::field_of(values.num_fields(), b);
    const std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>& hessians =
        values.physical_hessians(field);
    const Scalar viscosity = this->viscosity();

    // N_j ∂_b u_a − μ ∂_a ∂_b N_j, and for a = b also u · ∇N_j − μ ΔN_j
    Eigen::MatrixX<Scalar> result =
        values.values(field) * state.gradient[a][b].matrix().asDiagonal()
        - viscosity * hessians[pair_index<n>(a, b)];

    if (a == b) {
        result += convected(values.physical_gradients(field), state);

        for (std::size_t k = 0; k < n; ++k)
            result -= viscosity * hessians[pair_index<n>(k, k)];
    }

    return result;
}

template class NavierStokesElement<TensorBSpline<double, 2>, 2>;
template class NavierStokesElement<TensorBSpline<double, 3>, 3>;
template class NavierStokesElement<TensorNURBS<double, 2>, 2>;
template class NavierStokesElement<TensorNURBS<double, 3>, 3>;
template class NavierStokesElement<THBSpline<double, 2>, 2>;

} // namespace iguana
