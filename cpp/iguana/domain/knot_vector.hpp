/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_DOMAIN_KNOT_VECTOR_HPP
#define IGUANA_DOMAIN_KNOT_VECTOR_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Knot vector of a univariate spline space of a given degree
 *
 * With the degree, the knots fix the parametric domain, from the knot of
 * index p to the one of index num_knots - p - 1, and its elements, the
 * non-empty knot spans inside it
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
class KnotVector
{
public:
    /**
     * @brief Constructs the knot vector of a spline space
     *
     * @param degree Polynomial degree of the space
     * @param knots Non-decreasing knots
     *
     * @throws std::invalid_argument If the degree is negative, if the knots
     *         are not non-decreasing, if they number fewer than 2(p + 1),
     *         or if the parametric domain is empty
     */
    KnotVector(int degree, std::vector<T> knots);

    /// @brief Polynomial degree of the space
    constexpr int degree() const noexcept
    { return degree_; }

    /// @brief Non-decreasing knots
    constexpr const std::vector<T>& values() const noexcept
    { return knots_; }

    /// @brief Number of elements, the non-empty knot spans of the domain
    constexpr int num_elements() const noexcept
    { return static_cast<int>(element_spans_.size()); }

    /**
     * @brief Index of the knot at which an element starts, the last of its
     *        repeats, so that the next knot is larger
     *
     * @param element Element index
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr int element_span(int element) const noexcept
    { return element_spans_[static_cast<std::size_t>(element)]; }

    /**
     * @brief Parameter at which an element starts
     *
     * @param element Element index
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr T element_start(int element) const noexcept
    { return knots_[element_span(element)]; }

    /**
     * @brief Parameter at which an element ends
     *
     * @param element Element index
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr T element_end(int element) const noexcept
    { return knots_[element_span(element) + 1]; }

    /// @brief Parameter at which the parametric domain starts
    constexpr T domain_start() const noexcept
    { return element_start(0); }

    /// @brief Parameter at which the parametric domain ends
    constexpr T domain_end() const noexcept
    { return element_end(num_elements() - 1); }

private:
    /// @brief Polynomial degree of the space
    int degree_;

    /// @brief Non-decreasing knots
    std::vector<T> knots_;

    /// @brief Knot-span index of each element
    std::vector<int> element_spans_;
};

/**
 * @brief Knot vector with knots inserted, which refines its elements
 *
 * Each knot splits the element it falls in, or raises the multiplicity of
 * the knot it matches. The degree and the parametric domain stay the same,
 * so every function of a basis on the knot vector is a combination of the
 * functions of a basis on the new one, as refinement_matrix() gives them
 *
 * @param knot_vector Knot vector to refine
 * @param knots Knots to insert, in any order and possibly repeated
 * @return Knot vector of the same degree with the knots of both
 *
 * @throws std::invalid_argument If a knot lies outside the interior of the
 *         parametric domain
 */
template<std::floating_point T>
KnotVector<T> insert_knots(const KnotVector<T>& knot_vector,
                           const std::vector<T>& knots);

/**
 * @brief Coefficients of the functions of a spline space in a finer one
 *
 * The fine knot vector holds every knot of the coarse one, so each coarse
 * B-spline is a combination \f$ N_i = \sum_j R_{ji} M_j \f$ of the fine
 * ones. The coefficients follow from inserting the added knots one at a
 * time with Boehm's algorithm, as for the control points of a curve
 *
 * @param coarse Knot vector of the coarse space
 * @param fine Knot vector of the fine space, holding the knots of the
 *        coarse one and others
 * @return Matrix R with one row per fine function and one column per
 *         coarse function, whose column i holds the coefficients of coarse
 *         function i
 *
 * @throws std::invalid_argument If the degrees differ, if the fine knots
 *         do not contain the coarse ones, or if an added knot lies outside
 *         the interior of the parametric domain
 */
template<std::floating_point T>
Eigen::MatrixX<T> refinement_matrix(const KnotVector<T>& coarse,
                                    const KnotVector<T>& fine);

} // namespace iguana

#endif // IGUANA_DOMAIN_KNOT_VECTOR_HPP
