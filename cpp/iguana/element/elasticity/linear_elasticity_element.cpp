/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "linear_elasticity_element.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
LinearElasticityElement<Basis, n>::LinearElasticityElement(
    Scalar lame, Scalar shear_modulus)
    : lame_(lame), shear_modulus_(shear_modulus)
{
    if (shear_modulus <= 0)
        throw std::invalid_argument("LinearElasticityElement: "
                                    "the shear modulus must be positive");

    if (lame + 2 * shear_modulus / static_cast<Scalar>(n) <= 0)
        throw std::invalid_argument("LinearElasticityElement: "
                                    "the bulk modulus must be positive");
}

template<typename Basis, std::size_t n>
void LinearElasticityElement<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights, const Eigen::MatrixX<Scalar>&,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    const std::array<Eigen::MatrixX<Scalar>, n>& grads =
        values.physical_gradients();
    const Eigen::Index rows = grads[0].rows();
    const Eigen::Index size = static_cast<Eigen::Index>(n) * rows;

    tangent.resize(size, size);

    // Block (a, b) pairs component a of the test functions with component b
    // of the trial ones
    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(a) * rows;

        for (std::size_t b = 0; b < n; ++b) {
            const Eigen::Index col = static_cast<Eigen::Index>(b) * rows;

            // K_ij = ( 2μ ε(N_i e_a), ε(N_j e_b) )_Ω
            //        + ( λ div(N_i e_a), div(N_j e_b) )_Ω
            //      ≈ Σ_q w_q (μ δ_ab ∇N_i · ∇N_j + μ ∂_b N_i ∂_a N_j
            //                 + λ ∂_a N_i ∂_b N_j)
            tangent.block(row, col, rows, rows) =
                shear_modulus_ * grads[b] * weights.asDiagonal()
                    * grads[a].transpose()
                + lame_ * grads[a] * weights.asDiagonal()
                      * grads[b].transpose();

            if (a == b) {
                for (std::size_t c = 0; c < n; ++c)
                    tangent.block(row, col, rows, rows) +=
                        shear_modulus_ * grads[c] * weights.asDiagonal()
                        * grads[c].transpose();
            }
        }
    }
}

template<typename Basis, std::size_t n>
void LinearElasticityElement<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Eigen::MatrixX<Scalar> tangent;
    local_tangent(values, weights, source, coefficients, tangent);

    // R_i = Σ_j K_ij c_j, less the load
    residual = tangent * coefficients;

    const Eigen::MatrixX<Scalar>& functions = values.values();
    const Eigen::Index rows = functions.rows();

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index component = static_cast<Eigen::Index>(a);

        // F_i = ( N_i, f_a )_Ω
        //     ≈ Σ_q w_q f_a(x_q) N_i(x_q)
        residual.segment(component * rows, rows) -=
            functions * weights.cwiseProduct(source.col(component));
    }
}

template<typename Basis, std::size_t n>
void LinearElasticityElement<Basis, n>::local_mass(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    Eigen::MatrixX<Scalar>& mass) const
{
    const Eigen::MatrixX<Scalar>& functions = values.values();
    const Eigen::Index rows = functions.rows();
    const Eigen::Index size = static_cast<Eigen::Index>(n) * rows;

    mass.setZero(size, size);

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(a) * rows;

        // M_ij = ( N_i e_a, N_j e_a )_Ω
        //      ≈ Σ_q w_q N_i(x_q) N_j(x_q)
        mass.block(row, row, rows, rows) =
            functions * weights.asDiagonal() * functions.transpose();
    }
}

template class LinearElasticityElement<TensorBSpline<double, 2>, 2>;
template class LinearElasticityElement<TensorBSpline<double, 3>, 3>;
template class LinearElasticityElement<TensorNURBS<double, 2>, 2>;
template class LinearElasticityElement<TensorNURBS<double, 3>, 3>;

} // namespace iguana
