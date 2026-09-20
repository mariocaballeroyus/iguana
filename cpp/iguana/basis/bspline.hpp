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

    /// @brief Number of basis functions.
    constexpr int num_functions() const noexcept
    { return static_cast<int>(knots_.size()) - degree_ - 1; }

    /// @brief Number of functions active on each element.
    constexpr int num_active() const noexcept
    { return degree_ + 1; }

    /// @brief Number of non-empty knot spans in the parametric domain.
    constexpr int num_elements() const noexcept
    { return static_cast<int>(element_spans_.size()); }

    /**
     * @brief Index of the first function active on an element.
     *
     * @param element Element index.
     * @return Index of the first active basis function.
     *
     * @pre @p element lies in [0, num_elements()).
     */
    constexpr int first_active(int element) const noexcept
    { return element_spans_[static_cast<std::size_t>(element)] - degree_; }

    /**
     * @brief Parameter at which an element starts.
     *
     * @param element Element index.
     * @return Left endpoint of the element.
     *
     * @pre @p element lies in [0, num_elements()).
     */
    constexpr T element_start(int element) const noexcept
    { return knots_[element_spans_[element]]; }

    /**
     * @brief Parameter at which an element ends.
     *
     * @param element Element index.
     * @return Right endpoint of the element.
     *
     * @pre @p element lies in [0, num_elements()).
     */
    constexpr T element_end(int element) const noexcept
    { return knots_[element_spans_[element] + 1]; }

    /**
     * @brief Evaluates the non-zero functions on an element using the Cox-de
     *        Boor recursion.
     *
     * @param first_active Index of the first function active on the element.
     * @param points Parameters at which the functions are evaluated.
     * @param values Output matrix with size (num_active,num_points). It is
     *        resized when necessary.
     *
     * @pre @p first_active is the first active function of an existing,
     *      non-empty knot span.
     * @pre Every point in @p points lies inside that element.
     */
    void eval_on_element(int first_active, std::span<const T> points,
                         Eigen::MatrixX<T>& values) const;

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
