/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "skeleton_ghost_penalty.hpp"

#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
SkeletonGhostPenalty<Basis, n>::SkeletonGhostPenalty(
    const Trace<Basis, n>& trace, Scalar penalty, int scaling)
    : trace_(trace), penalty_(penalty), scaling_(scaling)
{
    if (penalty == 0)
        throw std::invalid_argument("SkeletonGhostPenalty: "
                                    "the penalty must not be zero");

    if (scaling < 0)
        throw std::invalid_argument("SkeletonGhostPenalty: "
                                    "the scaling must be non-negative");
}

template<typename Basis, std::size_t n>
void SkeletonGhostPenalty<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const Eigen::VectorX<Scalar>& weights, int order,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    // [[∂ₙᵖ T(N_i)]], jumps of the derivatives, after the face minus before
    Eigen::MatrixX<Scalar> before_trace;
    Eigen::MatrixX<Scalar> after_trace;
    trace_.local_trace(before, before_trace);
    trace_.local_trace(after, after_trace);
    Eigen::MatrixX<Scalar> jumps(before_trace.rows() + after_trace.rows(),
                                 before_trace.cols());
    jumps << -before_trace, after_trace;

    // Penalty times h^(2p−1+s), with h the mean size of the two cells
    const Scalar exponent = static_cast<Scalar>(2 * order - 1 + scaling_);
    const Eigen::ArrayX<Scalar> sizes = (before.sizes() + after.sizes()) / 2;
    const Eigen::VectorX<Scalar> penalized =
        (penalty_ * weights.array() * sizes.pow(exponent)).matrix();

    // K_ij = β ( [[∂ₙᵖ T(N_i)]], h^(2p−1+s) [[∂ₙᵖ T(N_j)]] )_Γ
    //      ≈ Σ_q w_q β h_q^(2p−1+s) J_iq J_jq
    tangent = jumps * penalized.asDiagonal() * jumps.transpose();
}

template<typename Basis, std::size_t n>
void SkeletonGhostPenalty<Basis, n>::local_residual(
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

template class SkeletonGhostPenalty<TensorBSpline<double, 2>, 2>;
template class SkeletonGhostPenalty<TensorBSpline<double, 2>, 3>;
template class SkeletonGhostPenalty<TensorBSpline<double, 3>, 3>;
template class SkeletonGhostPenalty<TensorNURBS<double, 2>, 2>;
template class SkeletonGhostPenalty<TensorNURBS<double, 2>, 3>;
template class SkeletonGhostPenalty<TensorNURBS<double, 3>, 3>;

} // namespace iguana
