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
void StokesStabilizedElement<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::MatrixX<Scalar>& tangent) const
{
    Base::local_tangent(values, weights, source, coefficients, tangent);
    add_stabilization(values, weights, tangent);
}

template<typename Basis, std::size_t n>
void StokesStabilizedElement<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Base::local_residual(values, weights, source, coefficients, residual);

    // R_i = Σ_j S_ij c_j over the stabilizing terms, less their load
    Eigen::MatrixX<Scalar> stabilization =
        Eigen::MatrixX<Scalar>::Zero(residual.size(), residual.size());
    add_stabilization(values, weights, stabilization);
    residual += stabilization * coefficients;

    const std::array<Eigen::MatrixX<Scalar>, n>& pressure_grads =
        values.physical_gradients(Base::field_of(values.num_fields(), n));
    const Eigen::Index pressure_row = Base::first_row(values, n);

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index component = static_cast<Eigen::Index>(a);

        // F_i = − ( τ_M ∇N_i, f )_Ω
        //     ≈ − Σ_q w_q τ_M ∇N_i(x_q) · f(x_q)
        residual.segment(pressure_row, pressure_grads[a].rows()) +=
            pressure_grads[a] * stabilized.cwiseProduct(source.col(component));
    }
}

template<typename Basis, std::size_t n>
void StokesStabilizedElement<Basis, n>::add_stabilization(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& tangent) const
{
    const std::size_t num_fields = values.num_fields();
    const std::array<Eigen::MatrixX<Scalar>, n>& pressure_grads =
        values.physical_gradients(Base::field_of(num_fields, n));
    const Eigen::Index pressure_row = Base::first_row(values, n);
    const Eigen::Index pressure_rows = pressure_grads[0].rows();
    const Scalar viscosity = this->viscosity();

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    for (std::size_t a = 0; a < n; ++a) {
        const std::array<Eigen::MatrixX<Scalar>, n>& test_grads =
            values.physical_gradients(Base::field_of(num_fields, a));
        const Eigen::Index row = Base::first_row(values, a);

        for (std::size_t b = 0; b < n; ++b) {
            const std::array<Eigen::MatrixX<Scalar>, n>& trial_grads =
                values.physical_gradients(Base::field_of(num_fields, b));
            const Eigen::Index col = Base::first_row(values, b);

            // K_ij = ( τ_C div(N_i e_a), div(N_j e_b) )_Ω
            //      ≈ Σ_q w_q μ ∂_a N_i ∂_b N_j
            tangent.block(row, col, test_grads[a].rows(),
                            trial_grads[b].rows()) +=
                viscosity * test_grads[a] * weights.asDiagonal()
                * trial_grads[b].transpose();
        }
    }

    for (std::size_t b = 0; b < n; ++b) {
        const std::size_t field = Base::field_of(num_fields, b);
        const std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>& hessians =
            values.physical_hessians(field);
        const Eigen::Index col = Base::first_row(values, b);
        const Eigen::Index cols = hessians[0].rows();

        // ΔN_j, and ∇N_i · ∇∂_b N_j from the gradient of the divergence
        Eigen::MatrixX<Scalar> laplacians =
            Eigen::MatrixX<Scalar>::Zero(cols, hessians[0].cols());
        Eigen::MatrixX<Scalar> curvature =
            Eigen::MatrixX<Scalar>::Zero(pressure_rows, cols);

        for (std::size_t a = 0; a < n; ++a) {
            laplacians += hessians[a];
            curvature += pressure_grads[a] * stabilized.asDiagonal()
                         * hessians[pair_index<n>(a, b)].transpose();
        }

        // K_ij = − ( τ_M ∇N_i, −div(2μ ε(N_j e_b)) )_Ω
        //      ≈ Σ_q w_q τ_M μ (∂_b N_i ΔN_j + ∇N_i · ∇∂_b N_j)
        tangent.block(pressure_row, col, pressure_rows, cols) +=
            viscosity * (pressure_grads[b] * stabilized.asDiagonal()
                             * laplacians.transpose()
                         + curvature);
    }

    // K_ij = − ( τ_M ∇N_i, ∇N_j )_Ω
    //      ≈ − Σ_q w_q τ_M ∇N_i · ∇N_j
    for (std::size_t c = 0; c < n; ++c) {
        tangent.block(pressure_row, pressure_row, pressure_rows,
                        pressure_rows) -=
            pressure_grads[c] * stabilized.asDiagonal()
            * pressure_grads[c].transpose();
    }
}

template<typename Basis, std::size_t n>
void StokesStabilizedElement<Basis, n>::local_mass(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& mass) const
{
    Base::local_mass(values, weights, mass);

    const std::size_t num_fields = values.num_fields();
    const std::array<Eigen::MatrixX<Scalar>, n>& pressure_grads =
        values.physical_gradients(Base::field_of(num_fields, n));
    const Eigen::Index pressure_row = Base::first_row(values, n);

    // Physical weights times τ_M
    const Eigen::VectorX<Scalar> stabilized =
        weights.cwiseProduct(tau(values));

    for (std::size_t b = 0; b < n; ++b) {
        const Eigen::MatrixX<Scalar>& functions =
            values.values(Base::field_of(num_fields, b));

        // M_ij = − ( τ_M ∇N_i, N_j e_b )_Ω
        //      ≈ − Σ_q w_q τ_M ∂_b N_i(x_q) N_j(x_q)
        mass.block(pressure_row, Base::first_row(values, b),
                   pressure_grads[b].rows(), functions.rows()) -=
            pressure_grads[b] * stabilized.asDiagonal()
            * functions.transpose();
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
