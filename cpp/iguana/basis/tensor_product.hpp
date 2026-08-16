/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_BASIS_TENSOR_PRODUCT_HPP
#define IGUANA_BASIS_TENSOR_PRODUCT_HPP

#include<array>
#include<concepts>

#include "iguana/basis/bspline.hpp"

namespace iguana
{

/**
 * @brief Tensor-product B-spline basis of a given parametric dimension.
 *
 * The basis is the product of one univariate basis per parametric
 * direction. Functions and elements alike are numbered lexicographically with 
 * the first direction running fastest.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the knot values.
 */
template<int d, std::floating_point T>
class TensorProductBSpline
{
    static_assert(d >= 1, "TensorProductBSpline: "
                          "the parametric dimension must be positive");

public:
    /// @brief The number of parametric directions, \f$ d \f$.
    static constexpr int dimension = d;

    /**
     * @brief Constructs the basis from one univariate basis per direction.
     *
     * @param axes The univariate bases, one per parametric direction.
     */
    explicit TensorProductBSpline(std::array<BSpline<T>, d> axes);

    /**
     * @brief The univariate basis of a given parametric direction.
     *
     * @param direction The direction index, in [0, dimension).
     *
     * @pre @p direction lies in [0, dimension). It is not checked.
     */
    constexpr const BSpline<T>& axis(int direction) const noexcept
    { return axes_[static_cast<std::size_t>(direction)]; }

    /// @brief The number of basis functions, over all the directions.
    constexpr int num_functions() const noexcept
    { return num_functions_; }

    /// @brief The number of elements, over all the directions.
    constexpr int num_elements() const noexcept
    { return num_elements_; }

    /// @brief The number of non-zero functions on any element, over all
    ///        the directions.
    constexpr int num_active() const noexcept
    { return num_active_; }

private:
    /// @brief The univariate bases, one per parametric direction.
    std::array<BSpline<T>, d> axes_;

    /// @brief The number of basis functions.
    int num_functions_;

    /// @brief The number of elements.
    int num_elements_;

    /// @brief The number of non-zero functions on any element.
    int num_active_;
};

} // namespace iguana

#endif // IGUANA_BASIS_TENSOR_PRODUCT_HPP
