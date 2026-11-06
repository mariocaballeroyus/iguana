/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "skeleton_multiplier_condition.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

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
 * @details The rows of the cell before the face, the + side, scaled by γ,
 *          are followed by those of the cell after, scaled by 1 − γ, so that
 *          a function active on both cells gets the mean once its rows are
 *          summed
 */
template<typename Scalar>
Eigen::MatrixX<Scalar> mean_rows(const Eigen::MatrixX<Scalar>& before,
                                 const Eigen::MatrixX<Scalar>& after,
                                 Scalar gamma)
{
    Eigen::MatrixX<Scalar> rows(before.rows() + after.rows(), before.cols());
    rows << gamma * before, (1 - gamma) * after;

    return rows;
}

/**
 * @brief Mean traces of the components of a vector along a direction at
 *        each point
 *
 *     {{T(N)}}_γ · d = Σ_c {{T_c(N)}}_γ d_c
 */
template<typename Basis, std::size_t n>
Eigen::MatrixX<typename Basis::Scalar> traces_along(
    const std::vector<std::reference_wrapper<const Trace<Basis, n>>>& traces,
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after, typename Basis::Scalar gamma,
    const std::type_identity_t<PointMatrix<typename Basis::Scalar, n>>&
        directions)
{
    using Scalar = typename Basis::Scalar;

    Eigen::MatrixX<Scalar> rows_before;
    Eigen::MatrixX<Scalar> rows_after;
    Eigen::MatrixX<Scalar> along;

    for (std::size_t c = 0; c < n; ++c) {
        traces[c].get().local_trace(before, rows_before);
        traces[c].get().local_trace(after, rows_after);
        const Eigen::MatrixX<Scalar> component =
            mean_rows(rows_before, rows_after, gamma)
            * directions.col(static_cast<Eigen::Index>(c)).asDiagonal();

        if (c == 0)
            along = component;
        else
            along += component;
    }

    return along;
}

/**
 * @brief Mean fluxes of the components of a vector along the normal of a
 *        face, taken along a direction at each point
 *
 *     {{F(N)}}_γ · d = Σ_c {{F_c(N)}}_γ d_c
 */
template<typename Basis, std::size_t n>
Eigen::MatrixX<typename Basis::Scalar> fluxes_along(
    const std::vector<std::reference_wrapper<const Flux<Basis, n>>>& fluxes,
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after, typename Basis::Scalar gamma,
    const std::type_identity_t<PointMatrix<typename Basis::Scalar, n>>&
        normals,
    const std::type_identity_t<PointMatrix<typename Basis::Scalar, n>>&
        directions)
{
    using Scalar = typename Basis::Scalar;

    Eigen::MatrixX<Scalar> rows_before;
    Eigen::MatrixX<Scalar> rows_after;
    Eigen::MatrixX<Scalar> along;

    for (std::size_t c = 0; c < n; ++c) {
        fluxes[c].get().local_flux(before, normals, rows_before);
        fluxes[c].get().local_flux(after, normals, rows_after);
        const Eigen::MatrixX<Scalar> component =
            mean_rows(rows_before, rows_after, gamma)
            * directions.col(static_cast<Eigen::Index>(c)).asDiagonal();

        if (c == 0)
            along = component;
        else
            along += component;
    }

    return along;
}

/// @brief Tangents of a planar boundary at the closest points of a range of
///        points, its normals there turned a quarter counterclockwise
template<typename Scalar>
PointMatrix<Scalar, 2> tangents_at(const BoundaryProjection<Scalar>& projection,
                                   int first, Eigen::Index count)
{
    PointMatrix<Scalar, 2> tangents(count, 2);
    tangents.col(0) = -projection.normals.col(1).segment(first, count);
    tangents.col(1) = projection.normals.col(0).segment(first, count);

    return tangents;
}

} // namespace

template<typename Basis, std::size_t n>
SkeletonMultiplierCondition<Basis, n>::SkeletonMultiplierCondition(
    Traces velocity, Fluxes fluxes, const Trace<Basis, n>& pressure,
    std::vector<BSpline<Scalar>> bases,
    const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order, Scalar penalty,
    bool symmetric)
    : velocity_(std::move(velocity)), fluxes_(std::move(fluxes)),
      pressure_(pressure), basis_(std::move(bases)), quadrature_(quadrature),
      projection_(projection), order_(order), penalty_(penalty),
      symmetric_(symmetric)
{
    if (velocity_.size() != n || fluxes_.size() != n)
        throw std::invalid_argument("SkeletonMultiplierCondition: "
                                    "the velocity needs a trace and a flux "
                                    "per component");

    if (order < 0)
        throw std::invalid_argument("SkeletonMultiplierCondition: "
                                    "the order must be non-negative");

    // Only the symmetric method needs a penalty to be stable
    if (symmetric && penalty <= 0)
        throw std::invalid_argument("SkeletonMultiplierCondition: "
                                    "the penalty must be positive");

    if (penalty < 0)
        throw std::invalid_argument("SkeletonMultiplierCondition: "
                                    "the penalty must be non-negative");

    basis_.check(projection_.faces, projection_.parameters);

    // A multiplier that no point reads would leave its equation empty
    std::vector<int> read;
    basis_.active_on(projection_.faces, projection_.parameters, 0,
                     static_cast<int>(projection_.faces.size()), read);

    if (std::ssize(read) != basis_.num_functions())
        throw std::invalid_argument("SkeletonMultiplierCondition: "
                                    "every multiplier must be read at a "
                                    "closest point");
}

template<typename Basis, std::size_t n>
ValueFlags SkeletonMultiplierCondition<Basis, n>::flags() const noexcept
{
    ValueFlags flags = pressure_.flags();

    for (std::size_t c = 0; c < n; ++c) {
        for (const ValueFlags read :
             {velocity_[c].get().flags(), fluxes_[c].get().flags()}) {
            flags.physical_gradients =
                flags.physical_gradients || read.physical_gradients;
            flags.physical_hessians =
                flags.physical_hessians || read.physical_hessians;
        }
    }

    return flags;
}

template<typename Basis, std::size_t n>
void SkeletonMultiplierCondition<Basis, n>::active_on(
    int first, int count, std::vector<int>& multipliers) const
{
    basis_.active_on(projection_.faces, projection_.parameters, first, count,
                     multipliers);
}

template<typename Basis, std::size_t n>
Eigen::MatrixX<typename Basis::Scalar>
SkeletonMultiplierCondition<Basis, n>::multiplier_values(int first,
                                                         int count) const
{
    std::vector<int> multipliers;
    active_on(first, count, multipliers);

    Eigen::MatrixX<Scalar> values =
        Eigen::MatrixX<Scalar>::Zero(std::ssize(multipliers), count);
    std::vector<int> at_point;
    Eigen::VectorX<Scalar> functions;

    for (int q = 0; q < count; ++q) {
        basis_.active_at(projection_.faces(first + q),
                         projection_.parameters(first + q), at_point,
                         functions);

        for (std::size_t k = 0; k < at_point.size(); ++k) {
            const auto row = std::ranges::lower_bound(multipliers,
                                                      at_point[k])
                             - multipliers.begin();
            values(row, q) = functions(static_cast<Eigen::Index>(k));
        }
    }

    return values;
}

template<typename Basis, std::size_t n>
void SkeletonMultiplierCondition<Basis, n>::local_tangent(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const ElementValues<Basis, n>& shifted_before,
    const ElementValues<Basis, n>& shifted_after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals, int first,
    Eigen::MatrixX<Scalar>& tangent) const
{
    // Jump of the weights along the normal, and γ of the weighted mean
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar gamma = cell_weights[0] / (cell_weights[0] + cell_weights[1]);
    const PointMatrix<Scalar, n> tangents =
        tangents_at(projection_, first, weights.size());

    // The sign s of the adjoint term, −1 in the symmetric method and +1 in
    // the skew-symmetric one
    const Scalar adjoint = symmetric_ ? -1 : 1;

    // − j ( v, {{2μ ε(u) n}}_γ )_F, the viscous flux of every component,
    // whose test functions are continuous across the face
    Eigen::MatrixX<Scalar> rows_before;
    Eigen::MatrixX<Scalar> rows_after;

    for (std::size_t c = 0; c < n; ++c) {
        velocity_[c].get().local_trace(before, rows_before);
        velocity_[c].get().local_trace(after, rows_after);
        const Eigen::MatrixX<Scalar> trace =
            mean_rows(rows_before, rows_after, gamma);

        fluxes_[c].get().local_flux(before, normals, rows_before);
        fluxes_[c].get().local_flux(after, normals, rows_after);
        const Eigen::MatrixX<Scalar> flux =
            mean_rows(rows_before, rows_after, gamma);

        if (c == 0)
            tangent.setZero(trace.rows(), trace.rows());

        tangent -= jump * trace * weights.asDiagonal() * flux.transpose();
    }

    // The tangential velocity by Nitsche's method, along τ
    const Eigen::MatrixX<Scalar> shifted =
        traces_along(velocity_, shifted_before, shifted_after, gamma,
                     tangents);
    const Eigen::MatrixX<Scalar> flux =
        fluxes_along(fluxes_, before, after, gamma, normals, tangents);

    // h = {{h}}, the weighted mean of the sizes of the two cells
    const Eigen::VectorX<Scalar> sizes =
        gamma * before.sizes() + (1 - gamma) * after.sizes();
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * std::abs(jump) * weights.cwiseQuotient(sizes);

    // K_ij += s j ( {{2μ ε(N_i) n}}_γ · τ, {{S N_j}}_γ · τ )_F
    //         + β |j| ( {{S N_i}}_γ · τ, {{S N_j}}_γ · τ / h )_F
    tangent += adjoint * jump * flux * weights.asDiagonal()
                   * shifted.transpose()
               + shifted * penalized.asDiagonal() * shifted.transpose();

    // K_ij −= j ( N_i · n, {{S P_j}}_γ − {{P_j}}_γ )_F, the pressure carried
    // from the closest points to the face, where the multipliers stand for
    // it
    const Eigen::MatrixX<Scalar> normal =
        traces_along(velocity_, before, after, gamma, normals);

    pressure_.local_trace(shifted_before, rows_before);
    pressure_.local_trace(shifted_after, rows_after);
    Eigen::MatrixX<Scalar> change = mean_rows(rows_before, rows_after, gamma);
    pressure_.local_trace(before, rows_before);
    pressure_.local_trace(after, rows_after);
    change -= mean_rows(rows_before, rows_after, gamma);

    tangent -= jump * normal * weights.asDiagonal() * change.transpose();
}

template<typename Basis, std::size_t n>
void SkeletonMultiplierCondition<Basis, n>::local_load(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const ElementValues<Basis, n>& shifted_before,
    const ElementValues<Basis, n>& shifted_after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals, const Eigen::MatrixX<Scalar>& data,
    int first, Eigen::VectorX<Scalar>& load) const
{
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar gamma = cell_weights[0] / (cell_weights[0] + cell_weights[1]);
    const PointMatrix<Scalar, n> tangents =
        tangents_at(projection_, first, weights.size());
    const Scalar adjoint = symmetric_ ? -1 : 1;

    const Eigen::MatrixX<Scalar> shifted =
        traces_along(velocity_, shifted_before, shifted_after, gamma,
                     tangents);
    const Eigen::MatrixX<Scalar> flux =
        fluxes_along(fluxes_, before, after, gamma, normals, tangents);
    const Eigen::VectorX<Scalar> sizes =
        gamma * before.sizes() + (1 - gamma) * after.sizes();
    const Eigen::VectorX<Scalar> penalized =
        penalty_ * std::abs(jump) * weights.cwiseQuotient(sizes);

    // g · τ, the tangential velocity imposed
    const Eigen::VectorX<Scalar> imposed =
        (data.array() * tangents.array()).rowwise().sum();

    // F_i = s j ( {{2μ ε(N_i) n}}_γ · τ, g · τ )_F
    //       + β |j| ( {{S N_i}}_γ · τ, g · τ / h )_F
    load = adjoint * jump * flux * weights.asDiagonal() * imposed
           + shifted * penalized.asDiagonal() * imposed;
}

template<typename Basis, std::size_t n>
void SkeletonMultiplierCondition<Basis, n>::local_coupling(
    const ElementValues<Basis, n>& before,
    const ElementValues<Basis, n>& after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals, int first,
    Eigen::MatrixX<Scalar>& coupling) const
{
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar gamma = cell_weights[0] / (cell_weights[0] + cell_weights[1]);
    const Eigen::MatrixX<Scalar> normal =
        traces_along(velocity_, before, after, gamma, normals);

    // C_kj = j ( μ_k ∘ s, N_j · n )_F
    //      ≈ Σ_q w_q j μ_k(s_q) (N_j · n)(x_q)
    coupling = jump
               * multiplier_values(first, static_cast<int>(weights.size()))
               * weights.asDiagonal() * normal.transpose();
}

template<typename Basis, std::size_t n>
void SkeletonMultiplierCondition<Basis, n>::local_constraint(
    const ElementValues<Basis, n>& shifted_before,
    const ElementValues<Basis, n>& shifted_after,
    const std::array<Scalar, 2>& cell_weights,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals, const Eigen::MatrixX<Scalar>& data,
    int first, Eigen::MatrixX<Scalar>& constraint,
    Eigen::VectorX<Scalar>& load) const
{
    const Scalar jump = cell_weights[0] - cell_weights[1];
    const Scalar gamma = cell_weights[0] / (cell_weights[0] + cell_weights[1]);
    const Eigen::MatrixX<Scalar> shifted =
        traces_along(velocity_, shifted_before, shifted_after, gamma,
                     normals);
    const Eigen::MatrixX<Scalar> tested =
        jump * multiplier_values(first, static_cast<int>(weights.size()))
        * weights.asDiagonal();

    // B_kj = j ( μ_k ∘ s, {{S N_j}}_γ · n )_F
    //      ≈ Σ_q w_q j μ_k(s_q) ({{S N_j}}_γ · n)(x_q)
    constraint = tested * shifted.transpose();

    // G_k = j ( μ_k ∘ s, g · n )_F
    load = tested * (data.array() * normals.array()).rowwise().sum().matrix();
}

template<typename Basis, std::size_t n>
Eigen::VectorX<typename Basis::Scalar>
SkeletonMultiplierCondition<Basis, n>::values(
    const Eigen::VectorX<Scalar>& multipliers) const
{
    if (multipliers.size() != num_multipliers())
        throw std::invalid_argument("SkeletonMultiplierCondition: "
                                    "the coefficients must be one per "
                                    "multiplier");

    // λ(s_q) = Σ_k λ_k μ_k(s_q)
    return basis_.evaluate(multipliers, projection_.faces,
                           projection_.parameters);
}

template class SkeletonMultiplierCondition<TensorBSpline<double, 2>, 2>;
template class SkeletonMultiplierCondition<TensorNURBS<double, 2>, 2>;

} // namespace iguana
