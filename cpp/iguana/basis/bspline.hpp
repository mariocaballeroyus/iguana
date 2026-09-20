/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_BASIS_BSPLINE_HPP
#define IGUANA_BASIS_BSPLINE_HPP

#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Univariate B-spline basis.
 *
 * The basis is defined by its polynomial degree and knot vector. Its elements
 * are the non-empty knot spans in the parametric domain.
 *
 * @tparam T Floating-point type.
 */
template<std::floating_point T>
class BSpline
{
public:
    /// @brief Largest degree supported by the basis algorithms.
    static constexpr int max_degree = 10;

    /**
     * @brief Constructs a univariate B-spline basis.
     *
     * @param degree Polynomial degree of the basis.
     * @param knots Non-decreasing knot vector.
     *
     * @throws std::invalid_argument If the @p degree is negative or above
     *         #max_degree, if @p knots is not non-decreasing, or if @p knots
     *         holds fewer than \f$ 2(p+1) \f$ entries, or if the parametric
     *         domain is empty.
     */
    BSpline(int degree, std::vector<T> knots);

    /// @brief Polynomial degree of the basis.
    constexpr int degree() const noexcept
    { return degree_; }

    /// @brief Non-decreasing knot vector.
    constexpr const std::vector<T>& knots() const noexcept
    { return knots_; }

private:
    /// @brief Polynomial degree of the basis.
    int degree_;

    /// @brief Non-decreasing knot vector.
    std::vector<T> knots_;

    /// @brief Knot-span index of each element.
    std::vector<int> element_spans_;
};

} // namespace iguana

#endif // IGUANA_BASIS_BSPLINE_HPP
