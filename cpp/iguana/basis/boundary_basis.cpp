/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary_basis.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "iguana/grid/knot_vector.hpp"

namespace iguana
{

template<std::floating_point T>
BoundaryBasis<T>::BoundaryBasis(std::vector<BSpline<T>> bases)
    : bases_(std::move(bases)), offsets_{0}
{
    for (const BSpline<T>& basis : bases_)
        offsets_.push_back(offsets_.back() + basis.num_functions());
}

template<std::floating_point T>
void BoundaryBasis<T>::check(const Eigen::VectorXi& faces,
                             const Eigen::VectorX<T>& parameters) const
{
    for (Eigen::Index point = 0; point < faces.size(); ++point) {
        const int face = faces(point);

        if (face < 0 || face >= static_cast<int>(bases_.size()))
            throw std::invalid_argument("BoundaryBasis: "
                                        "every face of the points needs a "
                                        "basis");

        // Up to round-off, the points lie in the interval of their basis
        const KnotVector<T>& knots =
            bases_[static_cast<std::size_t>(face)].knots();
        const T rounding = 4096 * std::numeric_limits<T>::epsilon()
                           * (knots.domain_end() - knots.domain_start());

        if (!(parameters(point) >= knots.domain_start() - rounding
              && parameters(point) <= knots.domain_end() + rounding))
            throw std::invalid_argument("BoundaryBasis: "
                                        "the basis of a face must cover the "
                                        "parameters of its points");
    }
}

template<std::floating_point T>
void BoundaryBasis<T>::active_at(int face, T parameter,
                                 std::vector<int>& functions,
                                 Eigen::VectorX<T>& values) const
{
    const BSpline<T>& basis = bases_[static_cast<std::size_t>(face)];
    const int first_active =
        basis.first_active(element_holding(basis.knots(), parameter));

    Eigen::MatrixX<T> at_point;
    basis.eval_on_element(first_active, std::span<const T>(&parameter, 1),
                          at_point);
    values = at_point.col(0);
    functions.resize(static_cast<std::size_t>(at_point.rows()));

    for (std::size_t k = 0; k < functions.size(); ++k)
        functions[k] = offsets_[static_cast<std::size_t>(face)]
                       + first_active + static_cast<int>(k);
}

template<std::floating_point T>
void BoundaryBasis<T>::active_on(const Eigen::VectorXi& faces,
                                 const Eigen::VectorX<T>& parameters,
                                 int first, int count,
                                 std::vector<int>& functions) const
{
    std::vector<int> at_point;
    Eigen::VectorX<T> values;
    functions.clear();

    for (int point = first; point < first + count; ++point) {
        active_at(faces(point), parameters(point), at_point, values);
        functions.insert(functions.end(), at_point.begin(), at_point.end());
    }

    std::sort(functions.begin(), functions.end());
    functions.erase(std::unique(functions.begin(), functions.end()),
                    functions.end());
}

template<std::floating_point T>
Eigen::VectorX<T> BoundaryBasis<T>::evaluate(
    const Eigen::VectorX<T>& coefficients, const Eigen::VectorXi& faces,
    const Eigen::VectorX<T>& parameters) const
{
    Eigen::VectorX<T> result(faces.size());
    std::vector<int> functions;
    Eigen::VectorX<T> values;

    for (Eigen::Index point = 0; point < result.size(); ++point) {
        active_at(faces(point), parameters(point), functions, values);

        // f(s_q) = Σ_i c_i N_i(s_q)
        result(point) = values.dot(coefficients(functions));
    }

    return result;
}

template class BoundaryBasis<double>;

} // namespace iguana
