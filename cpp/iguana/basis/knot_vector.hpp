/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_BASIS_KNOT_VECTOR_HPP
#define IGUANA_BASIS_KNOT_VECTOR_HPP

#include <concepts>
#include <cstddef>
#include <vector>

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

} // namespace iguana

#endif // IGUANA_BASIS_KNOT_VECTOR_HPP
