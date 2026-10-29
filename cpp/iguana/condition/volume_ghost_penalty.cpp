/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "volume_ghost_penalty.hpp"

#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
VolumeGhostPenalty<Basis, n>::VolumeGhostPenalty(
    const Trace<Basis, n>& trace, Scalar penalty, int scaling)
    : trace_(trace), penalty_(penalty), scaling_(scaling)
{
    if (penalty == 0)
        throw std::invalid_argument("VolumeGhostPenalty: "
                                    "the penalty must not be zero");

    if (scaling < 0)
        throw std::invalid_argument("VolumeGhostPenalty: "
                                    "the scaling must be non-negative");
}

template<typename Basis, std::size_t n>
void VolumeGhostPenalty<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after_in_before,
    const ElementValues<Basis, n>& before_in_after,
    const ElementValues<Basis, n>& after,
    const Eigen::VectorX<Scalar>& weights_before,
    const Eigen::VectorX<Scalar>& weights_after,
    const Eigen::VectorX<Scalar>&, Eigen::MatrixX<Scalar>& tangent) const
{
    // D(N_i), the polynomial of the cell before minus that of the cell
    // after, at the points of either, the rows of the cell before first
    Eigen::MatrixX<Scalar> first;
    Eigen::MatrixX<Scalar> second;
    trace_.local_trace(before, first);
    trace_.local_trace(after_in_before, second);
    Eigen::MatrixX<Scalar> in_before(first.rows() + second.rows(),
                                     first.cols());
    in_before << first, -second;

    trace_.local_trace(before_in_after, first);
    trace_.local_trace(after, second);
    Eigen::MatrixX<Scalar> in_after(first.rows() + second.rows(),
                                    first.cols());
    in_after << first, -second;

    // Penalty times h^(s−2), with h the mean size of the two cells
    const Scalar exponent = static_cast<Scalar>(scaling_ - 2);
    const Eigen::ArrayX<Scalar> sizes_before =
        (before.sizes() + after_in_before.sizes()) / 2;
    const Eigen::ArrayX<Scalar> sizes_after =
        (before_in_after.sizes() + after.sizes()) / 2;
    const Eigen::VectorX<Scalar> penalized_before =
        (penalty_ * weights_before.array() * sizes_before.pow(exponent))
            .matrix();
    const Eigen::VectorX<Scalar> penalized_after =
        (penalty_ * weights_after.array() * sizes_after.pow(exponent))
            .matrix();

    // K_ij = γ Σ_T ( D(N_i), h^(s−2) D(N_j) )_T
    //      ≈ Σ_T Σ_q w_q γ h_q^(s−2) D_iq D_jq
    tangent = in_before * penalized_before.asDiagonal()
                  * in_before.transpose()
              + in_after * penalized_after.asDiagonal()
                    * in_after.transpose();
}

template<typename Basis, std::size_t n>
void VolumeGhostPenalty<Basis, n>::local_residual(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after_in_before,
    const ElementValues<Basis, n>& before_in_after,
    const ElementValues<Basis, n>& after,
    const Eigen::VectorX<Scalar>& weights_before,
    const Eigen::VectorX<Scalar>& weights_after,
    const Eigen::VectorX<Scalar>& coefficients,
    Eigen::VectorX<Scalar>& residual) const
{
    Eigen::MatrixX<Scalar> tangent;
    local_tangent(before, after_in_before, before_in_after, after,
                  weights_before, weights_after, coefficients, tangent);

    // R_i = Σ_j K_ij c_j
    residual = tangent * coefficients;
}

template class VolumeGhostPenalty<TensorBSpline<double, 2>, 2>;
template class VolumeGhostPenalty<TensorBSpline<double, 2>, 3>;
template class VolumeGhostPenalty<TensorBSpline<double, 3>, 3>;
template class VolumeGhostPenalty<TensorNURBS<double, 2>, 2>;
template class VolumeGhostPenalty<TensorNURBS<double, 2>, 3>;
template class VolumeGhostPenalty<TensorNURBS<double, 3>, 3>;

} // namespace iguana
