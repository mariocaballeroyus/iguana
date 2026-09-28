/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "bspline.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iguana
{

template<std::floating_point T>
BSpline<T>::BSpline(int degree, std::vector<T> knots)
    : degree_(degree), knots_(std::move(knots))
{
    if (degree_ < 0)
        throw std::invalid_argument("BSpline: "
                                    "the degree must be non-negative");

    if (degree_ > max_degree)
        throw std::invalid_argument("BSpline: "
                                    "the degree exceeds max_degree");

    if (!std::ranges::is_sorted(knots_))
        throw std::invalid_argument("BSpline: "
                                    "the knots must be non-decreasing");

    const std::size_t min_knots =
        2 * (static_cast<std::size_t>(degree_) + 1);

    if (knots_.size() < min_knots)
        throw std::invalid_argument("BSpline: "
                                    "too few knots for the given degree");

    const int num_functions =
        static_cast<int>(knots_.size()) - degree_ - 1;

    for (int span = degree_; span < num_functions; ++span) {
        if (knots_[span] < knots_[span + 1])
            element_spans_.push_back(span);
    }

    if (element_spans_.empty())
        throw std::invalid_argument("BSpline: "
                                    "the parametric domain must be non-empty");
}

template<std::floating_point T>
void BSpline<T>::eval_on_element(int first_active, std::span<const T> points,
                                 Eigen::MatrixX<T>& values) const
{
    const int deg = degree_;
    const Eigen::Index num_pts = static_cast<Eigen::Index>(points.size());
    const T* const knots = knots_.data();

    // Knot span shared by all evaluation points
    const int span = first_active + deg;

    // Each column holds the active values at one point
    values.resize(deg + 1, num_pts);

    // Stack storage bounded by max_degree
    T left[max_degree + 1];
    T right[max_degree + 1];

    for (Eigen::Index q = 0; q < num_pts; ++q) {
        const T u = points[static_cast<std::size_t>(q)];
        T* const n = values.col(q).data();

        // Piecewise-constant starting value
        n[0] = T{1};

        // Build degrees 1 .. p in place
        for (int j = 1; j <= deg; ++j) {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;

            T saved = T{0};

            for (int r = 0; r < j; ++r) {
                // Cox-de Boor coefficient for the two adjacent functions
                const T temp = n[r] / (right[r + 1] + left[j - r]);

                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            // Remaining contribution belongs to the last active function
            n[j] = saved;
        }
    }
}

namespace
{

/// @brief Whether knots lie inside the parametric domain of a basis, off
///        its ends, where inserting them refines its elements
template<std::floating_point T>
bool inside_domain(const BSpline<T>& basis, const std::vector<T>& knots)
{
    const T start = basis.element_start(0);
    const T end = basis.element_end(basis.num_elements() - 1);

    return std::ranges::all_of(knots, [start, end](T knot) {
        return start < knot && knot < end;
    });
}

} // namespace

template<std::floating_point T>
BSpline<T> insert_knots(const BSpline<T>& basis,
                        const std::vector<T>& knots)
{
    if (!inside_domain(basis, knots))
        throw std::invalid_argument("insert_knots: "
                                    "the knots must lie inside the "
                                    "parametric domain");

    std::vector<T> merged = basis.knots();
    merged.insert(merged.end(), knots.begin(), knots.end());
    std::ranges::sort(merged);

    return BSpline<T>(basis.degree(), std::move(merged));
}

template<std::floating_point T>
Eigen::MatrixX<T> refinement_matrix(const BSpline<T>& coarse,
                                    const BSpline<T>& fine)
{
    const int degree = coarse.degree();

    if (fine.degree() != degree)
        throw std::invalid_argument("refinement_matrix: "
                                    "the bases must share their degree");

    if (!std::ranges::includes(fine.knots(), coarse.knots()))
        throw std::invalid_argument("refinement_matrix: "
                                    "the fine knots must contain the coarse "
                                    "ones");

    // Knots of the fine basis beyond those of the coarse one, repeats
    // included
    std::vector<T> added;
    std::ranges::set_difference(fine.knots(), coarse.knots(),
                                std::back_inserter(added));

    if (!inside_domain(coarse, added))
        throw std::invalid_argument("refinement_matrix: "
                                    "the added knots must lie inside the "
                                    "parametric domain");

    // Each column holds the coefficients of one coarse function, at first
    // in the coarse basis itself
    const int num_coarse = coarse.num_functions();
    Eigen::MatrixX<T> result = Eigen::MatrixX<T>::Identity(num_coarse,
                                                           num_coarse);
    std::vector<T> knots = coarse.knots();

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

template class BSpline<double>;

template BSpline<double> insert_knots(const BSpline<double>&,
                                      const std::vector<double>&);
template Eigen::MatrixXd refinement_matrix(const BSpline<double>&,
                                           const BSpline<double>&);

} // namespace iguana
