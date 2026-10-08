/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "ghost_penalty.hpp"

#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
GhostPenalty<Basis, n>::GhostPenalty(const Trace<Basis, n>& trace,
                                     Scalar penalty)
    : trace_(trace), penalty_(penalty)
{
    if (penalty <= 0)
        throw std::invalid_argument("GhostPenalty: "
                                    "the penalty must be positive");
}

template<typename Basis, std::size_t n>
void GhostPenalty<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const Eigen::VectorX<Scalar>& weights, int order,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    Eigen::MatrixX<Scalar> before_trace;
    Eigen::MatrixX<Scalar> after_trace;
    trace_.local_trace(before, before_trace);
    trace_.local_trace(after, after_trace);

    // Jump of each function across the face, the cell after minus the cell
    // before, with the functions of the cell before first
    Eigen::MatrixX<Scalar> jumps(before_trace.rows() + after_trace.rows(),
                                 before_trace.cols());
    jumps << -before_trace, after_trace;

    // h is the mean size of the two cells
    const Scalar exponent = static_cast<Scalar>(2 * order - 1);
    const Eigen::ArrayX<Scalar> sizes = (before.sizes() + after.sizes()) / 2;
    const Eigen::VectorX<Scalar> penalized =
        (penalty_ * weights.array() * sizes.pow(exponent)).matrix();

    // K_ij = γ ∫ h^(2p−1) [[∂ₙᵖ T(N_i)]] [[∂ₙᵖ T(N_j)]] ds
    //      ≈ Σ_q γ w_q h_q^(2p−1) J_iq J_jq
    stiffness = jumps * penalized.asDiagonal() * jumps.transpose();
}

template class GhostPenalty<TensorBSpline<double, 2>, 2>;
template class GhostPenalty<TensorBSpline<double, 2>, 3>;
template class GhostPenalty<TensorBSpline<double, 3>, 3>;
template class GhostPenalty<TensorNURBS<double, 2>, 2>;
template class GhostPenalty<TensorNURBS<double, 2>, 3>;
template class GhostPenalty<TensorNURBS<double, 3>, 3>;

} // namespace iguana
