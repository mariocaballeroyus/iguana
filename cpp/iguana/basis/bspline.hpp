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
#include<span>
#include<vector>

#include<Eigen/Core>

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
    /// @brief Largest degree, bounding the scratch of the evaluation.
    static constexpr int max_degree = 10;

    /**
     * @brief Constructs a univariate B-spline basis.
     *
     * @param degree The polynomial degree of the basis.
     * @param knots The non-decreasing sequence of knot values.
     *
     * @throws std::invalid_argument If @p degree is negative or above
     *         #max_degree, if @p knots is not non-decreasing, or if
     *         @p knots holds fewer than \f$ 2(p+1) \f$ entries.
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

    /**
     * @brief Evaluates the non-zero basis functions on a given element.
     *
     * Due to the local support property, only the \f$ p+1 \f$ consecutive
     * functions \f$ N_{a,p}, \ldots, N_{a+p,p} \f$ are non-zero. They are
     * evaluated using the Cox-de Boor recursion.
     *
     * @param first_active The index \f$ a \f$ of the first non-zero
     *        function on the element.
     * @param points The parameters at which the basis is evaluated.
     * @param values Output buffer of size (num_basis,num_points). Being
     *        column-major, each column holds the \f$ p+1 \f$ values of one
     *        point contiguously in memory. The buffer is resized if its shape
     *        changes.
     *
     * @pre @p first_active is the first non-zero function of an existing,
     *      non-empty element/span.
     * @pre Every point in @p points lies inside that element.
     *
     * Neither precondition is checked, and violating them reads outside
     * the knot vector.
     */
    void eval_on_element(int first_active, std::span<const T> points,
                         Eigen::MatrixX<T>& values) const;

private:
    /// @brief The polynomial degree of the basis.
    int degree_;

    /// @brief The non-decreasing sequence of knots.
    std::vector<T> knots_;
};

} // namespace iguana

#endif // IGUANA_BASIS_BSPLINE_HPP
