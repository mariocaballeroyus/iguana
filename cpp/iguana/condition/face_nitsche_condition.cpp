/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "face_nitsche_condition.hpp"

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
 * @brief Rows of a quantity over the functions of both cells of a face,
 *        those of the cell before scaled by its share of the weights and
 *        those of the cell after by the rest
 *
 * A function active on both cells gets the sum of its two rows, so that the
 * rows give the weighted mean {{w}} of the quantity
 */
template<typename Scalar>
Eigen::MatrixX<Scalar> mean_rows(const Eigen::MatrixX<Scalar>& before,
                                 const Eigen::MatrixX<Scalar>& after,
                                 Scalar share)
{
    Eigen::MatrixX<Scalar> rows(before.rows() + after.rows(), before.cols());
    rows << share * before, (1 - share) * after;

    return rows;
}

} // namespace

template<typename Basis, std::size_t n>
FaceNitscheCondition<Basis, n>::FaceNitscheCondition(
    const Trace<Basis, n>& trace, const Flux<Basis, n>& flux, Scalar penalty)
    : trace_(trace), flux_(flux), penalty_(penalty)
{
    if (penalty <= 0)
        throw std::invalid_argument("FaceNitscheCondition: "
                                    "the penalty must be positive");
}

template<typename Basis, std::size_t n>
void FaceNitscheCondition<Basis, n>::local_stiffness(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const ElementValues<Basis, n>& shifted_before,
    const ElementValues<Basis, n>& shifted_after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals,
    Eigen::MatrixX<Scalar>& stiffness) const
{
    // Jump of the weights along the normal, and the share of the cell before
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar share = cell_weights[0] / (cell_weights[0] + cell_weights[1]);

    Eigen::MatrixX<Scalar> before_rows;
    Eigen::MatrixX<Scalar> after_rows;

    trace_.local_trace(before, before_rows);
    trace_.local_trace(after, after_rows);
    const Eigen::MatrixX<Scalar> trace =
        mean_rows(before_rows, after_rows, share);

    flux_.local_flux(before, normals, before_rows);
    flux_.local_flux(after, normals, after_rows);
    const Eigen::MatrixX<Scalar> flux =
        mean_rows(before_rows, after_rows, share);

    trace_.local_trace(shifted_before, before_rows);
    trace_.local_trace(shifted_after, after_rows);
    const Eigen::MatrixX<Scalar> shifted =
        mean_rows(before_rows, after_rows, share);

    // h = {{h}}, the weighted mean of the sizes of the two cells
    const Eigen::VectorX<Scalar> sizes =
        share * before.sizes() + (1 - share) * after.sizes();
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * std::abs(jump) * weights.cwiseQuotient(sizes);

    // K_ij = − j ∫ T(N_i) {{F(N_j)}} ds − j ∫ {{F(N_i)}} {{T(S N_j)}} ds
    //        + β |j| ∫ {{T(S N_i)}} {{T(S N_j)}} / h ds
    //      ≈ Σ_q w_q (β |j| S_iq S_jq / h_q − j T_iq F_jq − j F_iq S_jq)
    stiffness =
        shifted * penalized.asDiagonal() * shifted.transpose()
        - jump * (trace * weights.asDiagonal() * flux.transpose()
                  + flux * weights.asDiagonal() * shifted.transpose());
}

template<typename Basis, std::size_t n>
void FaceNitscheCondition<Basis, n>::local_load(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const ElementValues<Basis, n>& shifted_before,
    const ElementValues<Basis, n>& shifted_after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals,
    const Eigen::VectorX<Scalar>& data, Eigen::VectorX<Scalar>& load) const
{
    // Jump of the weights along the normal, and the share of the cell before
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar share = cell_weights[0] / (cell_weights[0] + cell_weights[1]);

    Eigen::MatrixX<Scalar> before_rows;
    Eigen::MatrixX<Scalar> after_rows;

    flux_.local_flux(before, normals, before_rows);
    flux_.local_flux(after, normals, after_rows);
    const Eigen::MatrixX<Scalar> flux =
        mean_rows(before_rows, after_rows, share);

    trace_.local_trace(shifted_before, before_rows);
    trace_.local_trace(shifted_after, after_rows);
    const Eigen::MatrixX<Scalar> shifted =
        mean_rows(before_rows, after_rows, share);

    // h = {{h}}, the weighted mean of the sizes of the two cells
    const Eigen::VectorX<Scalar> sizes =
        share * before.sizes() + (1 - share) * after.sizes();
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * std::abs(jump) * weights.cwiseQuotient(sizes);

    // F_i = ∫ (β |j| {{T(S N_i)}} / h − j {{F(N_i)}}) g ds
    //     ≈ Σ_q w_q g(x_q) (β |j| S_iq / h_q − j F_iq)
    load = shifted * penalized.cwiseProduct(data)
           - jump * flux * weights.cwiseProduct(data);
}

template class FaceNitscheCondition<TensorBSpline<double, 2>, 2>;
template class FaceNitscheCondition<TensorBSpline<double, 2>, 3>;
template class FaceNitscheCondition<TensorBSpline<double, 3>, 3>;
template class FaceNitscheCondition<TensorNURBS<double, 2>, 2>;
template class FaceNitscheCondition<TensorNURBS<double, 2>, 3>;
template class FaceNitscheCondition<TensorNURBS<double, 3>, 3>;

} // namespace iguana
