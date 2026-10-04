/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "knot_vector.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iguana
{

template<std::floating_point T>
KnotVector<T>::KnotVector(int degree, std::vector<T> knots)
    : degree_(degree), knots_(std::move(knots))
{
    if (degree_ < 0)
        throw std::invalid_argument("KnotVector: "
                                    "the degree must be non-negative");

    if (!std::ranges::is_sorted(knots_))
        throw std::invalid_argument("KnotVector: "
                                    "the knots must be non-decreasing");

    const std::size_t min_knots =
        2 * (static_cast<std::size_t>(degree_) + 1);

    if (knots_.size() < min_knots)
        throw std::invalid_argument("KnotVector: "
                                    "too few knots for the given degree");

    // The parametric domain runs from the knot of index p to the one of
    // num_knots - p - 1, and its elements are the non-empty spans between
    const int last = static_cast<int>(knots_.size()) - degree_ - 1;

    for (int span = degree_; span < last; ++span) {
        if (knots_[span] < knots_[span + 1])
            element_spans_.push_back(span);
    }

    if (element_spans_.empty())
        throw std::invalid_argument("KnotVector: "
                                    "the parametric domain must be "
                                    "non-empty");
}

template<std::floating_point T>
KnotVector<T> insert_knots(const KnotVector<T>& knot_vector,
                           const std::vector<T>& knots)
{
    const T start = knot_vector.domain_start();
    const T end = knot_vector.domain_end();

    if (!std::ranges::all_of(knots, [start, end](T knot) {
            return start < knot && knot < end;
        }))
        throw std::invalid_argument("insert_knots: "
                                    "the knots must lie inside the "
                                    "parametric domain");

    std::vector<T> merged = knot_vector.values();
    merged.insert(merged.end(), knots.begin(), knots.end());
    std::ranges::sort(merged);

    return KnotVector<T>(knot_vector.degree(), std::move(merged));
}

template<std::floating_point T>
Eigen::MatrixX<T> refinement_matrix(const KnotVector<T>& coarse,
                                    const KnotVector<T>& fine)
{
    const int degree = coarse.degree();

    if (fine.degree() != degree)
        throw std::invalid_argument("refinement_matrix: "
                                    "the knot vectors must share their "
                                    "degree");

    if (!std::ranges::includes(fine.values(), coarse.values()))
        throw std::invalid_argument("refinement_matrix: "
                                    "the fine knots must contain the coarse "
                                    "ones");

    // Knots of the fine vector beyond those of the coarse one, repeats
    // included
    std::vector<T> added;
    std::ranges::set_difference(fine.values(), coarse.values(),
                                std::back_inserter(added));

    const T start = coarse.domain_start();
    const T end = coarse.domain_end();

    if (std::ranges::any_of(added, [start, end](T knot) {
            return knot <= start || knot >= end;
        }))
        throw std::invalid_argument("refinement_matrix: "
                                    "the added knots must lie inside the "
                                    "parametric domain");

    // Each column holds the coefficients of one coarse function, at first
    // in the coarse space itself
    const int num_coarse =
        static_cast<int>(coarse.values().size()) - degree - 1;
    Eigen::MatrixX<T> result = Eigen::MatrixX<T>::Identity(num_coarse,
                                                           num_coarse);
    std::vector<T> knots = coarse.values();

    for (const T knot : added) {
        // Span holding the knot, knots[span] <= knot < knots[span + 1]
        const int span =
            static_cast<int>(std::ranges::upper_bound(knots, knot)
                             - knots.begin())
            - 1;

        // The knot adds one function, and the rows follow as the control
        // points of a curve do. They are replaced from the last down, so
        // that each reads rows not yet replaced
        result.conservativeResize(result.rows() + 1, Eigen::NoChange);

        for (int function = static_cast<int>(result.rows()) - 1;
             function > span - degree; --function) {
            if (function > span) {
                // Past the knot, the functions shift by one
                result.row(function) = result.row(function - 1);
            }
            else {
                // Around the knot, each blends two
                const T alpha = (knot - knots[function])
                                / (knots[function + degree] - knots[function]);

                result.row(function) = alpha * result.row(function)
                                       + (T{1} - alpha)
                                             * result.row(function - 1);
            }
        }

        knots.insert(knots.begin() + span + 1, knot);
    }

    return result;
}

template<std::floating_point T>
std::vector<Eigen::MatrixX<T>> extraction_operators(
    const KnotVector<T>& knot_vector)
{
    const std::vector<T>& values = knot_vector.values();

    if (values.front() != knot_vector.domain_start()
        || values.back() != knot_vector.domain_end())
        throw std::invalid_argument("extraction_operators: "
                                    "the knot vector must be clamped");

    const int degree = knot_vector.degree();
    const int num_elements = knot_vector.num_elements();

    // Raise every interior knot to multiplicity p, so that the functions on
    // each element of the refined basis are its Bernstein polynomials
    std::vector<T> added;

    for (int element = 1; element < num_elements; ++element) {
        const T knot = knot_vector.element_start(element);
        const int multiplicity =
            static_cast<int>(std::ranges::count(values, knot));

        added.insert(added.end(), std::max(degree - multiplicity, 0), knot);
    }

    const KnotVector<T> bezier = insert_knots(knot_vector, added);
    const Eigen::MatrixX<T> refinement =
        refinement_matrix(knot_vector, bezier);

    // On each element, the block of the refinement matrix between the
    // Bernstein polynomials and the active functions, transposed
    std::vector<Eigen::MatrixX<T>> result;

    for (int element = 0; element < num_elements; ++element) {
        const int fine = bezier.element_span(element) - degree;
        const int coarse = knot_vector.element_span(element) - degree;

        result.push_back(
            refinement.block(fine, coarse, degree + 1, degree + 1)
                .transpose());
    }

    return result;
}

template class KnotVector<double>;

template KnotVector<double> insert_knots(const KnotVector<double>&,
                                         const std::vector<double>&);
template Eigen::MatrixXd refinement_matrix(const KnotVector<double>&,
                                           const KnotVector<double>&);
template std::vector<Eigen::MatrixXd> extraction_operators(
    const KnotVector<double>&);

} // namespace iguana
