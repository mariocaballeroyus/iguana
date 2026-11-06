/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "lagrange_multiplier_condition.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

namespace iguana
{

template<typename Basis, std::size_t n>
LagrangeMultiplierCondition<Basis, n>::LagrangeMultiplierCondition(
    const Trace<Basis, n>& trace, std::vector<BSpline<Scalar>> bases,
    const QuadratureBoundary<Scalar, dim>& quadrature)
    : traces_{std::cref(trace)}, basis_(std::move(bases)),
      quadrature_(quadrature)
{
    basis_.check(quadrature_.faces(), quadrature_.parameters());
}

template<typename Basis, std::size_t n>
LagrangeMultiplierCondition<Basis, n>::LagrangeMultiplierCondition(
    std::vector<std::reference_wrapper<const Trace<Basis, n>>> components,
    std::vector<BSpline<Scalar>> bases,
    const QuadratureBoundary<Scalar, dim>& quadrature)
    : traces_(std::move(components)), basis_(std::move(bases)),
      quadrature_(quadrature)
{
    if (traces_.size() != n)
        throw std::invalid_argument("LagrangeMultiplierCondition: "
                                    "the normal component needs a trace per "
                                    "component");

    basis_.check(quadrature_.faces(), quadrature_.parameters());
}

template<typename Basis, std::size_t n>
ValueFlags LagrangeMultiplierCondition<Basis, n>::flags() const noexcept
{
    ValueFlags flags;

    for (const Trace<Basis, n>& trace : traces_) {
        flags.physical_gradients = flags.physical_gradients
                                   || trace.flags().physical_gradients;
        flags.physical_hessians = flags.physical_hessians
                                  || trace.flags().physical_hessians;
    }

    return flags;
}

template<typename Basis, std::size_t n>
void LagrangeMultiplierCondition<Basis, n>::active_on(
    int first, int count, std::vector<int>& multipliers) const
{
    basis_.active_on(quadrature_.faces(), quadrature_.parameters(), first,
                     count, multipliers);
}

template<typename Basis, std::size_t n>
void LagrangeMultiplierCondition<Basis, n>::local_coupling(
    const ElementValues<Basis, n>& values,
    const Eigen::VectorX<Scalar>& weights,
    const PointMatrix<Scalar, n>& normals, int first,
    Eigen::MatrixX<Scalar>& coupling) const
{
    const int count = static_cast<int>(weights.size());

    // B_jq, the trace of each degree of freedom at the points, along the
    // normal for the components of a vector trace
    Eigen::MatrixX<Scalar> trace;
    traces_.front().get().local_trace(values, trace);

    if (normal()) {
        trace = trace * normals.col(0).asDiagonal();
        Eigen::MatrixX<Scalar> component;

        for (std::size_t c = 1; c < n; ++c) {
            traces_[c].get().local_trace(values, component);
            trace += component * normals.col(c).asDiagonal();
        }
    }

    std::vector<int> multipliers;
    active_on(first, count, multipliers);
    coupling.setZero(std::ssize(multipliers), trace.rows());

    std::vector<int> at_point;
    Eigen::VectorX<Scalar> functions;

    for (int q = 0; q < count; ++q) {
        basis_.active_at(quadrature_.faces()(first + q),
                         quadrature_.parameters()(first + q), at_point,
                         functions);

        for (std::size_t k = 0; k < at_point.size(); ++k) {
            const auto row = std::ranges::lower_bound(multipliers,
                                                      at_point[k])
                             - multipliers.begin();

            // B_ij = ( μ_i, T(N_j) )_Γ
            //      ≈ Σ_q w_q μ_i(s_q) B_jq
            coupling.row(row) += weights(q)
                                 * functions(static_cast<Eigen::Index>(k))
                                 * trace.col(q).transpose();
        }
    }
}

template<typename Basis, std::size_t n>
void LagrangeMultiplierCondition<Basis, n>::local_load(
    const Eigen::VectorX<Scalar>& weights, const Eigen::VectorX<Scalar>& data,
    int first, Eigen::VectorX<Scalar>& load) const
{
    const int count = static_cast<int>(weights.size());

    std::vector<int> multipliers;
    active_on(first, count, multipliers);
    load.setZero(std::ssize(multipliers));

    std::vector<int> at_point;
    Eigen::VectorX<Scalar> functions;

    for (int q = 0; q < count; ++q) {
        basis_.active_at(quadrature_.faces()(first + q),
                         quadrature_.parameters()(first + q), at_point,
                         functions);

        for (std::size_t k = 0; k < at_point.size(); ++k) {
            const auto row = std::ranges::lower_bound(multipliers,
                                                      at_point[k])
                             - multipliers.begin();

            // G_i = ( μ_i, g )_Γ
            //     ≈ Σ_q w_q μ_i(s_q) g(x_q)
            load(row) += weights(q) * functions(static_cast<Eigen::Index>(k))
                         * data(q);
        }
    }
}

template<typename Basis, std::size_t n>
Eigen::VectorX<typename Basis::Scalar>
LagrangeMultiplierCondition<Basis, n>::values(
    const Eigen::VectorX<Scalar>& multipliers) const
{
    if (multipliers.size() != num_multipliers())
        throw std::invalid_argument("LagrangeMultiplierCondition: "
                                    "the coefficients must be one per "
                                    "multiplier");

    // λ(s_q) = Σ_i λ_i μ_i(s_q)
    return basis_.evaluate(multipliers, quadrature_.faces(),
                           quadrature_.parameters());
}

template class LagrangeMultiplierCondition<TensorBSpline<double, 2>, 2>;
template class LagrangeMultiplierCondition<TensorNURBS<double, 2>, 2>;

} // namespace iguana
