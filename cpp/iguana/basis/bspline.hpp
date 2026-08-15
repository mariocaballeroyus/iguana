/*                                                 __ \/_     
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\  
 * This file is part of the IGUANA library.         \ \\ 
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_BASIS_BSPLINE_HPP
#define IGUANA_BASIS_BSPLINE_HPP

#include<concepts>
#include<vector>

namespace iguana
{

/**
 * @brief Univariate B-spline basis.
 *
 * The convention \f$ m = n + p + 1 \f$ is followed, where \f$ m \f$ is the 
 * number of knots, \f$ n \f$ the number of basis functions and \f$ p \f$ the 
 * polynomial degree.
 *
 * @tparam T Floating-point type of the knot values.
 */
template<std::floating_point T>
class BSpline
{
public:
    /**
     * @brief Constructs a univariate B-spline basis.
     * 
     * @param degree The polynomial degree of the basis.
     * @param knots The non-decreasing sequence of knot values.
     * 
     * @throws std::invalid_argument If @p degree is negative, if @p knots is 
     *         not non-decreasing, or if @p knots holds fewer than 
     *         \f$ 2(p+1) \f$ entries.
     */
    BSpline(int degree, std::vector<T> knots);

    /// @brief The polynomial degree of the basis, \f$ p \f$.
    constexpr int degree() const noexcept 
    { return degree_; }

    /// @brief The non-decreasing sequence of knots.
    constexpr const std::vector<T>& knots() const noexcept
    { return knots_; }

    /// @brief The number of knots, \f$ m \f$.
    constexpr int num_knots() const noexcept
    { return static_cast<int>(knots_.size()); }

    /// @brief The number of basis functions, \f$ n = m - p - 1 \f$.
    constexpr int num_functions() const noexcept
    { return static_cast<int>(knots_.size()) - degree_ - 1; }

private:
    /// @brief The polynomial degree of the basis.
    int degree_;

    /// @brief The non-decreasing sequence of knots.
    std::vector<T> knots_;
};

} // namespace iguana

#endif // IGUANA_BASIS_BSPLINE_HPP
