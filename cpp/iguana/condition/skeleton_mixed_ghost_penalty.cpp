/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "skeleton_mixed_ghost_penalty.hpp"

#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Derived divergence and derived pressure of the functions of one
 *        cell, a row per function of every field and a column per point
 *
 * @details The rows of the velocity hold the terms of the divergence, each
 *          in those of its own component, and zero pressure; the rows of the
 *          pressure hold zero divergence and the derived pressure
 */
template<typename Basis, std::size_t n>
void divergence_and_pressure(
    const ElementValues<Basis, n>& values,
    Eigen::MatrixX<typename Basis::Scalar>& divergence,
    Eigen::MatrixX<typename Basis::Scalar>& pressure)
{
    Eigen::Index rows = 0;

    for (std::size_t field = 0; field <= n; ++field)
        rows += values.values(field).rows();

    const Eigen::Index points = values.values(0).cols();
    divergence.setZero(rows, points);
    pressure.setZero(rows, points);

    Eigen::Index row = 0;

    for (std::size_t field = 0; field < n; ++field) {
        divergence.middleRows(row, values.values(field).rows()) =
            values.values(field);
        row += values.values(field).rows();
    }

    pressure.middleRows(row, values.values(n).rows()) = values.values(n);
}

} // namespace

template<typename Basis, std::size_t n>
SkeletonMixedGhostPenalty<Basis, n>::SkeletonMixedGhostPenalty(Scalar penalty)
    : penalty_(penalty)
{
    if (!(penalty > 0))
        throw std::invalid_argument("SkeletonMixedGhostPenalty: "
                                    "the penalty must be positive");
}

template<typename Basis, std::size_t n>
void SkeletonMixedGhostPenalty<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const Eigen::VectorX<Scalar>& weights, int order,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    // [[∂ₙᵏ div N_i]] and [[∂ₙᵏ N_i]], after the face minus before
    Eigen::MatrixX<Scalar> before_divergence;
    Eigen::MatrixX<Scalar> before_pressure;
    Eigen::MatrixX<Scalar> after_divergence;
    Eigen::MatrixX<Scalar> after_pressure;
    divergence_and_pressure(before, before_divergence, before_pressure);
    divergence_and_pressure(after, after_divergence, after_pressure);

    const Eigen::Index rows =
        before_divergence.rows() + after_divergence.rows();
    Eigen::MatrixX<Scalar> divergence(rows, weights.size());
    Eigen::MatrixX<Scalar> pressure(rows, weights.size());
    divergence << -before_divergence, after_divergence;
    pressure << -before_pressure, after_pressure;

    // Penalty times h^(2k + 1), with h the mean size of the two cells
    const Eigen::ArrayX<Scalar> sizes = (before.sizes() + after.sizes()) / 2;
    const Scalar exponent = static_cast<Scalar>(2 * order + 1);
    const Eigen::VectorX<Scalar> penalized =
        (penalty_ * weights.array() * sizes.pow(exponent)).matrix();

    // K_ij = − β ( [[∂ₙᵏ div N_i]], h^(2k+1) [[∂ₙᵏ N_j]] )_F
    //        − β ( [[∂ₙᵏ N_i]], h^(2k+1) [[∂ₙᵏ div N_j]] )_F
    //      ≈ − Σ_q w_q β h_q^(2k+1) (D_iq P_jq + P_iq D_jq)
    tangent = -divergence * penalized.asDiagonal() * pressure.transpose();
    tangent -= pressure * penalized.asDiagonal() * divergence.transpose();
}

template<typename Basis, std::size_t n>
void SkeletonMixedGhostPenalty<Basis, n>::local_residual(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const Eigen::VectorX<Scalar>& weights, int order,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Eigen::MatrixX<Scalar> tangent;
    local_tangent(before, after, weights, order, coefficients, tangent);

    // R_i = Σ_j K_ij c_j
    residual = tangent * coefficients;
}

template class SkeletonMixedGhostPenalty<TensorBSpline<double, 2>, 2>;
template class SkeletonMixedGhostPenalty<TensorNURBS<double, 2>, 2>;

} // namespace iguana
