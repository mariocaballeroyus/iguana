/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_PATCH_PATCH_HPP
#define IGUANA_PATCH_PATCH_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"

namespace iguana
{

/**
 * @brief Tensor-product B-spline patch, a map from the parameter box into
 *        physical space
 *
 * The patch pairs a basis with one control point per basis function, so that
 * a parameter maps to physical space, which is fixed to three dimensions
 *
 * @tparam T Floating-point type of the control points
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class Patch
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dim = d;

    /**
     * @brief Constructs the patch from a basis and its control points
     *
     * @param basis Tensor-product basis of the map
     * @param coefficients Control points, of size (num_functions,3), one row
     *        per basis function in the numbering of the basis
     *
     * @throws std::invalid_argument If the rows do not match the number of
     *         basis functions
     */
    Patch(TensorBSpline<T, d> basis, Eigen::MatrixX3<T> coefficients);

    /// @brief Basis of the map
    constexpr const TensorBSpline<T, d>& basis() const noexcept
    { return basis_; }

    /// @brief Control points, of size (num_functions,3)
    constexpr const Eigen::MatrixX3<T>& coefficients() const noexcept
    { return coefficients_; }

    /**
     * @brief Position of points of an element in physical space
     *
     * The positions are the control points of the active functions weighted
     * by the function values, so that the basis evaluation of the caller is
     * reused rather than repeated
     *
     * @param actives Functions that are non-zero on the element, as given by
     *        active_on_element()
     * @param values Their values at the points, of size
     *        (num_active,num_points), as given by eval_on_element()
     * @param positions Output buffer of size (num_points,3), resized if its
     *        shape changes
     *
     * @pre @p actives and @p values come from the same element. It is not
     *      checked
     */
    void position_on_element(const Eigen::VectorXi& actives,
                             const Eigen::MatrixX<T>& values,
                             Eigen::MatrixX3<T>& positions) const;

private:
    /// @brief Basis of the map
    TensorBSpline<T, d> basis_;

    /// @brief Control points, one row per basis function
    Eigen::MatrixX3<T> coefficients_;
};

} // namespace iguana

#endif // IGUANA_PATCH_PATCH_HPP
