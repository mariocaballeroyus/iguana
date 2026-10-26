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

namespace iguana
{

template<typename Basis, std::size_t n>
StokesElement<Basis, n>::StokesElement(Scalar viscosity)
    : viscosity_(viscosity)
{
    if (viscosity <= 0)
        throw std::invalid_argument("StokesElement: "
                                    "the viscosity must be positive");
}

template<typename Basis, std::size_t n>
void StokesElement<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights, const Eigen::MatrixX<Scalar>&,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    const std::size_t num_fields = values.num_fields();
    const Eigen::MatrixX<Scalar>& pressure =
        values.values(field_of(num_fields, n));
    const Eigen::Index pressure_row = first_row(values, n);
    const Eigen::Index size = first_row(values, n + 1);

    tangent.setZero(size, size);

    // Block (a, b) pairs component a of the test functions with component b
    // of the trial ones, the pressure following the velocity
    for (std::size_t a = 0; a < n; ++a) {
        const std::array<Eigen::MatrixX<Scalar>, n>& test_grads =
            values.physical_gradients(field_of(num_fields, a));
        const Eigen::Index row = first_row(values, a);
        const Eigen::Index rows = test_grads[a].rows();

        for (std::size_t b = 0; b < n; ++b) {
            const std::array<Eigen::MatrixX<Scalar>, n>& trial_grads =
                values.physical_gradients(field_of(num_fields, b));
            const Eigen::Index col = first_row(values, b);
            const Eigen::Index cols = trial_grads[b].rows();

            // K_ij = ( 2μ ε(N_i e_a), ε(N_j e_b) )_Ω
            //      ≈ Σ_q w_q μ (δ_ab ∇N_i · ∇N_j + ∂_b N_i ∂_a N_j)
            tangent.block(row, col, rows, cols) =
                viscosity_ * test_grads[b] * weights.asDiagonal()
                * trial_grads[a].transpose();

            if (a == b) {
                for (std::size_t c = 0; c < n; ++c)
                    tangent.block(row, col, rows, cols) +=
                        viscosity_ * test_grads[c] * weights.asDiagonal()
                        * trial_grads[c].transpose();
            }
        }

        // K_ij = − ( div(N_i e_a), N_j )_Ω
        //      ≈ − Σ_q w_q ∂_a N_i N_j
        tangent.block(row, pressure_row, rows, pressure.rows()) =
            -test_grads[a] * weights.asDiagonal() * pressure.transpose();
    }

    for (std::size_t b = 0; b < n; ++b) {
        const std::array<Eigen::MatrixX<Scalar>, n>& trial_grads =
            values.physical_gradients(field_of(num_fields, b));
        const Eigen::Index col = first_row(values, b);

        // K_ij = − ( N_i, div(N_j e_b) )_Ω
        //      ≈ − Σ_q w_q N_i ∂_b N_j
        tangent.block(pressure_row, col, pressure.rows(),
                        trial_grads[b].rows()) =
            -pressure * weights.asDiagonal() * trial_grads[b].transpose();
    }
}

template<typename Basis, std::size_t n>
void StokesElement<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    // The Galerkin tangent itself, not that of a derived element
    Eigen::MatrixX<Scalar> tangent;
    StokesElement::local_tangent(values, weights, source, coefficients,
                                 tangent);

    // R_i = Σ_j K_ij c_j, less the load of the velocity
    residual = tangent * coefficients;

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::MatrixX<Scalar>& functions =
            values.values(field_of(values.num_fields(), a));
        const Eigen::Index component = static_cast<Eigen::Index>(a);

        // F_i = ( N_i, f_a )_Ω
        //     ≈ Σ_q w_q f_a(x_q) N_i(x_q)
        residual.segment(first_row(values, a), functions.rows()) -=
            functions * weights.cwiseProduct(source.col(component));
    }
}

template<typename Basis, std::size_t n>
void StokesElement<Basis, n>::local_mass(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& mass) const
{
    const Eigen::Index size = first_row(values, n + 1);

    mass.setZero(size, size);

    for (std::size_t b = 0; b < n; ++b) {
        const Eigen::MatrixX<Scalar>& functions =
            values.values(field_of(values.num_fields(), b));
        const Eigen::Index row = first_row(values, b);

        // M_ij = ( N_i e_b, N_j e_b )_Ω
        //      ≈ Σ_q w_q N_i(x_q) N_j(x_q)
        mass.block(row, row, functions.rows(), functions.rows()) =
            functions * weights.asDiagonal() * functions.transpose();
    }
}

template class StokesElement<TensorBSpline<double, 2>, 2>;
template class StokesElement<TensorBSpline<double, 3>, 3>;
template class StokesElement<TensorNURBS<double, 2>, 2>;
template class StokesElement<TensorNURBS<double, 3>, 3>;

} // namespace iguana
