/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_BASIS_TENSOR_BSPLINE_HPP
#define IGUANA_BASIS_TENSOR_BSPLINE_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "bspline.hpp"

namespace iguana
{

/**
 * @brief Tensor-product B-spline basis.
 *
 * The basis is the product of one univariate B-spline basis per parametric
 * direction.
 *
 * @tparam T Floating-point type.
 * @tparam d Number of parametric directions.
 */
template<std::floating_point T, std::size_t d>
class TensorBSpline
{
    static_assert(d > 0, "TensorBSpline: "
                         "the parametric dimension must be positive");

public:
    /// @brief Number of parametric directions.
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs a tensor-product B-spline basis.
     *
     * @param axes Univariate basis of each parametric direction.
     */
    explicit TensorBSpline(std::array<BSpline<T>, d> axes);

    /**
     * @brief Univariate basis of a parametric direction.
     *
     * @param direction Direction index.
     * @return Basis associated with the direction.
     *
     * @pre @p direction lies in [0, dimension).
     */
    constexpr const BSpline<T>& axis(std::size_t direction) const noexcept
    { return axes_[direction]; }

    /// @brief Number of tensor-product basis functions.
    constexpr int num_functions() const noexcept
    { return num_functions_; }

    /// @brief Number of tensor-product elements.
    constexpr int num_elements() const noexcept
    { return num_elements_; }

    /// @brief Number of functions active on each element.
    constexpr int num_active() const noexcept
    { return num_active_; }

    /**
     * @brief Functions that are non-zero on an element.
     *
     * The returned indices follow the tensor-product numbering, with the
     * first parametric direction running fastest.
     *
     * @param element Element index.
     * @param actives Output vector of num_active() function indices. It is
     *        resized when necessary.
     *
     * @pre @p element lies in [0, num_elements()).
     */
    void active_on_element(int element, Eigen::VectorXi& actives) const;

private:
    /// @brief Univariate bases, one per parametric direction.
    std::array<BSpline<T>, d> axes_;

    /// @brief Product of the univariate function counts.
    int num_functions_;

    /// @brief Product of the univariate element counts.
    int num_elements_;

    /// @brief Product of the univariate active-function counts.
    int num_active_;
};

} // namespace iguana

#endif // IGUANA_BASIS_TENSOR_BSPLINE_HPP
