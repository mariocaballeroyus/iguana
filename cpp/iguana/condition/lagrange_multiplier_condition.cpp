/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "lagrange_multiplier_condition.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/grid/knot_vector.hpp"

namespace iguana
{

namespace
{

/// @brief First multiplier of each face, numbered face after face, followed
///        by their number
template<std::floating_point T>
std::vector<int> first_multipliers(const std::vector<BSpline<T>>& bases)
{
    std::vector<int> offsets{0};

    for (const BSpline<T>& basis : bases)
        offsets.push_back(offsets.back() + basis.num_functions());

    return offsets;
}

} // namespace

template<typename Basis, std::size_t n>
LagrangeMultiplierCondition<Basis, n>::LagrangeMultiplierCondition(
    const Trace<Basis, n>& trace, std::vector<BSpline<Scalar>> bases,
    const QuadratureBoundary<Scalar, dim>& quadrature)
    : traces_{std::cref(trace)}, bases_(std::move(bases)),
      offsets_(first_multipliers(bases_)), quadrature_(quadrature)
{
    check_bases();
}

template<typename Basis, std::size_t n>
LagrangeMultiplierCondition<Basis, n>::LagrangeMultiplierCondition(
    std::vector<std::reference_wrapper<const Trace<Basis, n>>> components,
    std::vector<BSpline<Scalar>> bases,
    const QuadratureBoundary<Scalar, dim>& quadrature)
    : traces_(std::move(components)), bases_(std::move(bases)),
      offsets_(first_multipliers(bases_)), quadrature_(quadrature)
{
    if (traces_.size() != n)
        throw std::invalid_argument("LagrangeMultiplierCondition: "
                                    "the normal component needs a trace per "
                                    "component");

    check_bases();
}

template<typename Basis, std::size_t n>
void LagrangeMultiplierCondition<Basis, n>::check_bases() const
{
    for (Eigen::Index point = 0; point < quadrature_.num_points(); ++point) {
        const int face = quadrature_.faces()(point);

        if (face < 0 || face >= static_cast<int>(bases_.size()))
            throw std::invalid_argument("LagrangeMultiplierCondition: "
                                        "every face of the boundary needs a "
                                        "basis");

        // Up to round-off, the points lie in the interval of their basis
        const KnotVector<Scalar>& knots =
            bases_[static_cast<std::size_t>(face)].knots();
        const Scalar parameter = quadrature_.parameters()(point);
        const Scalar rounding = 4096 * std::numeric_limits<Scalar>::epsilon()
                                * (knots.domain_end() - knots.domain_start());

        if (!(parameter >= knots.domain_start() - rounding
              && parameter <= knots.domain_end() + rounding))
            throw std::invalid_argument("LagrangeMultiplierCondition: "
                                        "the basis of a face must cover its "
                                        "parameter");
    }
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
void LagrangeMultiplierCondition<Basis, n>::active_at(
    int point, std::vector<int>& multipliers,
    Eigen::VectorX<Scalar>& functions) const
{
    const int face = quadrature_.faces()(point);
    const BSpline<Scalar>& basis = bases_[static_cast<std::size_t>(face)];
    const Scalar parameter = quadrature_.parameters()(point);
    const int first_active =
        basis.first_active(element_holding(basis.knots(), parameter));

    Eigen::MatrixX<Scalar> values;
    basis.eval_on_element(first_active,
                          std::span<const Scalar>(&parameter, 1), values);
    functions = values.col(0);

    multipliers.resize(static_cast<std::size_t>(values.rows()));

    for (std::size_t k = 0; k < multipliers.size(); ++k)
        multipliers[k] = offsets_[static_cast<std::size_t>(face)]
                         + first_active + static_cast<int>(k);
}

template<typename Basis, std::size_t n>
void LagrangeMultiplierCondition<Basis, n>::active_on(
    int first, int count, std::vector<int>& multipliers) const
{
    std::vector<int> at_point;
    Eigen::VectorX<Scalar> functions;
    multipliers.clear();

    for (int point = first; point < first + count; ++point) {
        active_at(point, at_point, functions);
        multipliers.insert(multipliers.end(), at_point.begin(),
                           at_point.end());
    }

    std::sort(multipliers.begin(), multipliers.end());
    multipliers.erase(std::unique(multipliers.begin(), multipliers.end()),
                      multipliers.end());
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
        active_at(first + q, at_point, functions);

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
        active_at(first + q, at_point, functions);

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

    Eigen::VectorX<Scalar> result(quadrature_.num_points());
    std::vector<int> at_point;
    Eigen::VectorX<Scalar> functions;

    for (Eigen::Index point = 0; point < result.size(); ++point) {
        active_at(static_cast<int>(point), at_point, functions);

        // λ(s_q) = Σ_i λ_i μ_i(s_q)
        result(point) = functions.dot(multipliers(at_point));
    }

    return result;
}

template class LagrangeMultiplierCondition<TensorBSpline<double, 2>, 2>;
template class LagrangeMultiplierCondition<TensorNURBS<double, 2>, 2>;

} // namespace iguana
