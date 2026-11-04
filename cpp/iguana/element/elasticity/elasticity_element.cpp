/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "elasticity_element.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Deformation gradient F = I + ∇u at a point, from the coefficients of
 *        the displacement, one component after another
 *
 * @param grads Gradients of the active functions, one matrix per coordinate
 * @param coefficients Coefficients on the element
 * @param point Index of the point
 * @return F, with a row per component and a column per coordinate
 */
template<typename Scalar, std::size_t n>
Eigen::Matrix<Scalar, n, n> deformation_gradient(
    const std::array<Eigen::MatrixX<Scalar>, n>& grads,
    const Eigen::VectorX<Scalar>& coefficients, Eigen::Index point)
{
    const Eigen::Index rows = grads[0].rows();
    Eigen::Matrix<Scalar, n, n> gradient =
        Eigen::Matrix<Scalar, n, n>::Identity();

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index row = static_cast<Eigen::Index>(a) * rows;

        for (std::size_t J = 0; J < n; ++J)
            gradient(a, J) +=
                coefficients.segment(row, rows).dot(grads[J].col(point));
    }

    return gradient;
}

} // namespace

template<typename Basis, std::size_t n>
ElasticityElement<Basis, n>::ElasticityElement(
    Scalar lame, Scalar shear_modulus)
    : lame_(lame), shear_modulus_(shear_modulus)
{
    if (shear_modulus <= 0)
        throw std::invalid_argument("ElasticityElement: "
                                    "the shear modulus must be positive");

    if (lame + 2 * shear_modulus / static_cast<Scalar>(n) <= 0)
        throw std::invalid_argument("ElasticityElement: "
                                    "the bulk modulus must be positive");
}

template<typename Basis, std::size_t n>
Eigen::Matrix<typename Basis::Scalar, n, n>
ElasticityElement<Basis, n>::stress(
    const Eigen::Matrix<Scalar, n, n>& gradient) const
{
    using Tensor = Eigen::Matrix<Scalar, n, n>;

    const Tensor strain =
        (gradient.transpose() * gradient - Tensor::Identity()) / 2;

    return 2 * shear_modulus_ * strain
           + lame_ * strain.trace() * Tensor::Identity();
}

template<typename Basis, std::size_t n>
void ElasticityElement<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights, const Eigen::MatrixX<Scalar>&,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::MatrixX<Scalar>& tangent) const
{
    using Tensor = Eigen::Matrix<Scalar, n, n>;

    const std::array<Eigen::MatrixX<Scalar>, n>& grads =
        values.physical_gradients();
    const Eigen::Index rows = grads[0].rows();
    const Eigen::Index size = static_cast<Eigen::Index>(n) * rows;

    tangent.setZero(size, size);

    // The gradients of the active functions at a point, a column per
    // coordinate
    Eigen::Matrix<Scalar, Eigen::Dynamic, n> point_grads(rows, n);

    for (Eigen::Index q = 0; q < weights.size(); ++q) {
        const Tensor gradient =
            deformation_gradient<Scalar, n>(grads, coefficients, q);
        const Tensor second_stress = stress(gradient);
        const Tensor left_stretch = gradient * gradient.transpose();

        for (std::size_t J = 0; J < n; ++J)
            point_grads.col(J) = grads[J].col(q);

        // Block (a, b) pairs component a of the test functions with component
        // b of the trial ones
        for (std::size_t a = 0; a < n; ++a) {
            const Eigen::Index row = static_cast<Eigen::Index>(a) * rows;

            for (std::size_t b = 0; b < n; ++b) {
                const Eigen::Index col = static_cast<Eigen::Index>(b) * rows;

                // K_ij = ( ∂P/∂F : ∇(N_j e_b), ∇(N_i e_a) )_Ω
                //      ≈ Σ_q w_q ∂_J N_i D_JL ∂_L N_j
                // D_JL = δ_ab S_JL + λ F_aJ F_bL + μ (F Fᵀ)_ab δ_JL
                //        + μ F_aL F_bJ
                Tensor moduli =
                    lame_ * gradient.row(a).transpose() * gradient.row(b)
                    + shear_modulus_ * left_stretch(a, b) * Tensor::Identity()
                    + shear_modulus_ * gradient.row(b).transpose()
                          * gradient.row(a);

                if (a == b)
                    moduli += second_stress;

                tangent.block(row, col, rows, rows) +=
                    weights(q) * point_grads * moduli
                    * point_grads.transpose();
            }
        }
    }
}

template<typename Basis, std::size_t n>
void ElasticityElement<Basis, n>::local_residual(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    using Tensor = Eigen::Matrix<Scalar, n, n>;

    const std::array<Eigen::MatrixX<Scalar>, n>& grads =
        values.physical_gradients();
    const Eigen::MatrixX<Scalar>& functions = values.values();
    const Eigen::Index rows = functions.rows();

    residual.setZero(static_cast<Eigen::Index>(n) * rows);

    for (Eigen::Index q = 0; q < weights.size(); ++q) {
        const Tensor gradient =
            deformation_gradient<Scalar, n>(grads, coefficients, q);

        // P = F S, the first Piola–Kirchhoff stress
        const Tensor first_stress = gradient * stress(gradient);

        for (std::size_t a = 0; a < n; ++a) {
            const Eigen::Index row = static_cast<Eigen::Index>(a) * rows;

            // R_i = ( P, ∇(N_i e_a) )_Ω
            //     ≈ Σ_q w_q P_aJ ∂_J N_i
            for (std::size_t J = 0; J < n; ++J)
                residual.segment(row, rows) +=
                    weights(q) * first_stress(a, J) * grads[J].col(q);
        }
    }

    for (std::size_t a = 0; a < n; ++a) {
        const Eigen::Index component = static_cast<Eigen::Index>(a);

        // F_i = ( N_i, f_a )_Ω
        //     ≈ Σ_q w_q f_a(x_q) N_i(x_q)
        residual.segment(component * rows, rows) -=
            functions * weights.cwiseProduct(source.col(component));
    }
}

template<typename Basis, std::size_t n>
void ElasticityElement<Basis, n>::local_mass(
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

template class ElasticityElement<TensorBSpline<double, 2>, 2>;
template class ElasticityElement<TensorBSpline<double, 3>, 3>;
template class ElasticityElement<TensorNURBS<double, 2>, 2>;
template class ElasticityElement<TensorNURBS<double, 3>, 3>;

} // namespace iguana
