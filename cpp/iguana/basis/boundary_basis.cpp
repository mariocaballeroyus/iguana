/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary_basis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "iguana/grid/knot_vector.hpp"

namespace iguana
{

template<std::floating_point T>
BoundaryBasis<T>::BoundaryBasis(std::vector<BSpline<T>> bases,
                                std::vector<bool> periodic)
    : bases_(std::move(bases)), periodic_(std::move(periodic)), offsets_{0}
{
    if (periodic_.empty())
        periodic_.assign(bases_.size(), false);

    if (periodic_.size() != bases_.size())
        throw std::invalid_argument("BoundaryBasis: "
                                    "there must be one periodic flag per "
                                    "face");

    for (std::size_t face = 0; face < bases_.size(); ++face) {
        const BSpline<T>& basis = bases_[face];
        const int degree = basis.degree();

        if (!periodic_[face]) {
            offsets_.push_back(offsets_.back() + basis.num_functions());
            continue;
        }

        // Wrapped, the last p functions are the first p ones again, and
        // those active on an element must stay distinct
        const int period = basis.num_functions() - degree;

        if (period <= degree)
            throw std::invalid_argument("BoundaryBasis: "
                                        "a periodic basis must keep more "
                                        "functions than its degree");

        // The spans before the interval and the first ones inside it
        // repeat one period later
        const std::vector<T>& t = basis.knots().values();
        const T rounding = 4096 * std::numeric_limits<T>::epsilon()
                           * (t.back() - t.front());

        for (int span = 0; span < 2 * degree; ++span) {
            const std::size_t i = static_cast<std::size_t>(span);
            const std::size_t j = static_cast<std::size_t>(span + period);

            if (std::abs((t[i + 1] - t[i]) - (t[j + 1] - t[j])) > rounding)
                throw std::invalid_argument("BoundaryBasis: "
                                            "the spans of a periodic basis "
                                            "must repeat after one period");
        }

        offsets_.push_back(offsets_.back() + period);
    }
}

template<std::floating_point T>
Eigen::VectorX<T> BoundaryBasis<T>::integrals() const
{
    Eigen::VectorX<T> result(num_functions());

    for (std::size_t face = 0; face < bases_.size(); ++face) {
        const std::vector<T>& t = bases_[face].knots().values();
        const int degree = bases_[face].degree();
        const int count = offsets_[face + 1] - offsets_[face];

        for (int function = 0; function < count; ++function) {
            // The whole of a function inside the interval, the one a period
            // later for those that wrap around
            const bool wraps = periodic_[face] && function < degree;
            const std::size_t i =
                static_cast<std::size_t>(wraps ? function + count : function);
            const std::size_t p = static_cast<std::size_t>(degree);

            // ∫ N_i = (t_{i+p+1} − t_i) / (p + 1)
            result(offsets_[face] + function) =
                (t[i + p + 1] - t[i]) / static_cast<T>(degree + 1);
        }
    }

    return result;
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

    // A periodic face numbers its functions modulo its count
    const std::size_t index = static_cast<std::size_t>(face);
    const int count = offsets_[index + 1] - offsets_[index];

    for (std::size_t k = 0; k < functions.size(); ++k) {
        const int function = first_active + static_cast<int>(k);
        functions[k] = offsets_[index]
                       + (periodic_[index] ? function % count : function);
    }
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
