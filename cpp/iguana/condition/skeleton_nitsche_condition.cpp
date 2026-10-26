/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "skeleton_nitsche_condition.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Weighted mean over the two cells of a face of a quantity of their
 *        functions
 *
 *     γ = α_{+} / (α_{+} + α_{-})
 *     {{.}}_γ = γ (.)_{+} + (1 − γ) (.)_{-}
 *
 * @details @p gamma is the share of the cell before the face, the + side,
 *          in the weights of the two cells. The rows of the cell before,
 *          scaled by γ, are followed by those of the cell after, scaled by
 *          1 − γ, so that a function active on both cells gets the mean once
 *          its two rows are summed
 */
template<typename Scalar>
Eigen::MatrixX<Scalar> mean_operator(const Eigen::MatrixX<Scalar>& before,
                                     const Eigen::MatrixX<Scalar>& after,
                                     Scalar gamma)
{
    Eigen::MatrixX<Scalar> rows(before.rows() + after.rows(), before.cols());
    rows << gamma * before, (1 - gamma) * after;

    return rows;
}

} // namespace

template<typename Basis, std::size_t n>
SkeletonNitscheCondition<Basis, n>::SkeletonNitscheCondition(
    const Trace<Basis, n>& trace, const Flux<Basis, n>& flux, Scalar penalty)
    : trace_(trace), flux_(flux), penalty_(penalty)
{
    if (penalty <= 0)
        throw std::invalid_argument("SkeletonNitscheCondition: "
                                    "the penalty must be positive");
}

template<typename Basis, std::size_t n>
void SkeletonNitscheCondition<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const ElementValues<Basis, n>& shifted_before,
    const ElementValues<Basis, n>& shifted_after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals, const Eigen::VectorX<Scalar>&,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    // Jump of the weights along the normal, and γ of the weighted mean
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar gamma = cell_weights[0] / (cell_weights[0] + cell_weights[1]);

    // Rows of each cell, reused for every quantity
    Eigen::MatrixX<Scalar> before_rows;
    Eigen::MatrixX<Scalar> after_rows;

    // T(N_i), the traces of the test functions, continuous across the face,
    // so that their mean is the traces themselves
    trace_.local_trace(before, before_rows);
    trace_.local_trace(after, after_rows);
    const Eigen::MatrixX<Scalar> trace =
        mean_operator(before_rows, after_rows, gamma);

    // {{F(N_i)}}_γ, mean fluxes along the normal of the functions of each cell
    flux_.local_flux(before, normals, before_rows);
    flux_.local_flux(after, normals, after_rows);
    const Eigen::MatrixX<Scalar> flux =
        mean_operator(before_rows, after_rows, gamma);

    // {{T(S N_i)}}_γ, mean traces of the functions of each cell
    trace_.local_trace(shifted_before, before_rows);
    trace_.local_trace(shifted_after, after_rows);
    const Eigen::MatrixX<Scalar> shifted =
        mean_operator(before_rows, after_rows, gamma);

    // h = {{h}}, the weighted mean of the sizes of the two cells
    const Eigen::VectorX<Scalar> sizes =
        gamma * before.sizes() + (1 - gamma) * after.sizes();
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * std::abs(jump) * weights.cwiseQuotient(sizes);

    // K_ij = − j ( T(N_i), {{F(N_j)}}_γ )_Γ
    //        − j ( {{F(N_i)}}_γ, {{T(S N_j)}}_γ )_Γ
    //        + β |j| ( {{T(S N_i)}}_γ, {{T(S N_j)}}_γ / h )_Γ
    //      ≈ Σ_q w_q (β |j| S_iq S_jq / h_q − j T_iq F_jq − j F_iq S_jq)
    tangent =
        shifted * penalized.asDiagonal() * shifted.transpose()
        - jump * (trace * weights.asDiagonal() * flux.transpose()
                  + flux * weights.asDiagonal() * shifted.transpose());
}

template<typename Basis, std::size_t n>
void SkeletonNitscheCondition<Basis, n>::local_residual(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const ElementValues<Basis, n>& shifted_before,
    const ElementValues<Basis, n>& shifted_after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Eigen::MatrixX<Scalar> tangent;
    local_tangent(before, after, shifted_before, shifted_after, cell_weights,
                  weights, normals, data, coefficients, tangent);

    // Jump of the weights along the normal, and γ of the weighted mean
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar gamma = cell_weights[0] / (cell_weights[0] + cell_weights[1]);

    // Rows of each cell, reused for every quantity
    Eigen::MatrixX<Scalar> before_rows;
    Eigen::MatrixX<Scalar> after_rows;

    // {{F(N_i)}}_γ, mean fluxes along the normal of the functions of each cell
    flux_.local_flux(before, normals, before_rows);
    flux_.local_flux(after, normals, after_rows);
    const Eigen::MatrixX<Scalar> flux =
        mean_operator(before_rows, after_rows, gamma);

    // {{T(S N_i)}}_γ, mean traces of the functions of each cell
    trace_.local_trace(shifted_before, before_rows);
    trace_.local_trace(shifted_after, after_rows);
    const Eigen::MatrixX<Scalar> shifted =
        mean_operator(before_rows, after_rows, gamma);

    // h = {{h}}, the weighted mean of the sizes of the two cells
    const Eigen::VectorX<Scalar> sizes =
        gamma * before.sizes() + (1 - gamma) * after.sizes();
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * std::abs(jump) * weights.cwiseQuotient(sizes);

    // F_i = β |j| ( {{T(S N_i)}}_γ / h, g )_Γ − j ( {{F(N_i)}}_γ, g )_Γ
    //     ≈ Σ_q w_q g(x_q) (β |j| S_iq / h_q − j F_iq)
    const Eigen::VectorX<Scalar> load =
        shifted * penalized.cwiseProduct(data)
        - jump * flux * weights.cwiseProduct(data);

    // R_i = Σ_j K_ij c_j − F_i
    residual = tangent * coefficients - load;
}

template class SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>;
template class SkeletonNitscheCondition<TensorBSpline<double, 2>, 3>;
template class SkeletonNitscheCondition<TensorBSpline<double, 3>, 3>;
template class SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>;
template class SkeletonNitscheCondition<TensorNURBS<double, 2>, 3>;
template class SkeletonNitscheCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
