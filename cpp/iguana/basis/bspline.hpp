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

    /// @brief The number of non-zero functions on any element, \f$ p+1 \f$.
    constexpr int num_active() const noexcept
    { return degree_ + 1; }

    /// @brief The number of elements (non-empty spans) of the domain.
    constexpr int num_elements() const noexcept
    { return num_elements_; }

    /**
     * @brief The first non-zero function on a given element.
     *
     * An element activates the num_active() consecutive functions that
     * start at the returned index.
     *
     * @param element The element index, in [0, num_elements()).
     *
     * @return The index to be passed as their first_active argument.
     *
     * @pre @p element lies in [0, num_elements()).
     *
     * The precondition is not checked, and violating it reads outside
     * the multiplicity sums.
     */
    constexpr int first_active(int element) const noexcept
    { return mult_sum_[first_element_ + element] - 1 - degree_; }

    /**
     * @brief The parameter at which a given element starts.
     *
     * @param element The element index, in [0, num_elements()).
     *
     * @pre @p element lies in [0, num_elements()). It is not checked.
     */
    constexpr T element_start(int element) const noexcept
    { return knots_[mult_sum_[first_element_ + element] - 1]; }

    /**
     * @brief The parameter at which a given element ends.
     *
     * @param element The element index, in [0, num_elements()).
     *
     * @pre @p element lies in [0, num_elements()). It is not checked.
     */
    constexpr T element_end(int element) const noexcept
    { return knots_[mult_sum_[first_element_ + element]]; }

    /**
     * @brief The element a parameter lies in.
     *
     * Parameters on an interior element interface belong to the later
     * element, and the end of the domain to the last one.
     *
     * @param point The parameter, inside the domain of the basis.
     *
     * @return The element index, in [0, num_elements()).
     *
     * @pre @p point lies inside the domain. It is not checked.
     */
    int element_at(T point) const noexcept;

    /**
     * @brief Inserts a knot, refining the basis.
     *
     * The knot joins the sequence, so that the span it lies in is split
     * and one function is gained. The refined basis spans every function
     * of the original one.
     *
     * @param knot The knot value, strictly inside the domain.
     *
     * @return The span the knot is inserted in, \f$ k \f$ such that
     *         \f$ \xi_k \le \bar{\xi} < \xi_{k+1} \f$ in the knots before
     *         the insertion.
     *
     * @throws std::invalid_argument If @p knot lies outside the domain
     *         or on its ends, or if its multiplicity would exceed the
     *         degree.
     */
    int insert_knot(T knot);

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
     * No precondition is checked, and violating them reads outside
     * the knot vector.
     */
    void eval_on_element(int first_active, std::span<const T> points,
                         Eigen::MatrixX<T>& values) const;

    /**
     * @brief Evaluates the non-zero basis functions on a given element,
     *        together with their derivatives up to a given order.
     *
     * The same recursion as eval_on_element is used, but its triangle is
     * kept and differenced to obtain the derivatives.
     *
     * @param first_active The index \f$ a \f$ of the first non-zero
     *        function on the element.
     * @param points The parameters at which the basis is evaluated.
     * @param order The highest derivative order to evaluate. It is 
     *        safe-guarded against orders higher than the basis degree.
     * @param values One output buffer per order, resized to @p order + 1.
     *        Entry \f$ k \f$ holds the derivatives of order \f$ k \f$, of
     *        size (num_basis,num_points). Being column-major, each column
     *        holds the \f$ p+1 \f$ values of one point contiguously in
     *        memory. Each buffer is resized if its shape changes.
     *
     * @pre @p first_active is the first non-zero function of an existing,
     *      non-empty element/span.
     * @pre Every point in @p points lies inside that element.
     * @pre @p order is non-negative.
     *
     * No precondition is checked, and violating them reads outside the
     * knot vector.
     */
    void eval_ders_on_element(int first_active, std::span<const T> points,
                              int order,
                              std::vector<Eigen::MatrixX<T>>& values) const;

private:
    /// @brief The polynomial degree of the basis.
    int degree_;

    /// @brief The non-decreasing sequence of knots.
    std::vector<T> knots_;

    /// @brief One entry per distinct knot value, holding the number of
    ///        knots up to and including it.
    std::vector<int> mult_sum_;

    /// @brief The number of distinct knot values before the first element.
    int first_element_;

    /// @brief The number of elements of the domain.
    int num_elements_;
};

} // namespace iguana

#endif // IGUANA_BASIS_BSPLINE_HPP
