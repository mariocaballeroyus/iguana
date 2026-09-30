/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_PATCH_HPP
#define IGUANA_GEOMETRY_PATCH_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"

namespace iguana
{

/**
 * @brief Matrix of points in physical space, one per row. The storage is
 *        row-major, so that a point is contiguous
 *
 * @tparam T Floating-point type of the coordinates
 * @tparam n Dimension of the physical space
 */
template<std::floating_point T, std::size_t n>
using PointMatrix = Eigen::Matrix<T, Eigen::Dynamic, n, Eigen::RowMajor>;

/**
 * @brief Tensor-product B-spline patch, a map from the parameter box into
 *        physical space
 *
 * The patch pairs a basis with one control point per basis function, so that
 * a parameter maps to physical space. A physical space of more dimensions
 * than parametric directions makes the patch a curve or surface in it, such
 * as a shell, and one of as many a domain, such as a planar region
 *
 * @tparam T Floating-point type of the control points
 * @tparam d Number of parametric directions
 * @tparam n Dimension of the physical space, at least d
 */
template<std::floating_point T, std::size_t d, std::size_t n>
class Patch
{
    static_assert(d <= n, "Patch: "
                          "the physical space needs a dimension per "
                          "parametric direction");

public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dim = d;

    /**
     * @brief Constructs the patch from a basis and its control points
     *
     * @param basis Tensor-product basis of the map
     * @param coefficients Control points, of size (num_functions,n), one row
     *        per basis function in the numbering of the basis
     *
     * @throws std::invalid_argument If the rows do not match the number of
     *         basis functions
     */
    Patch(TensorBSpline<T, d> basis, PointMatrix<T, n> coefficients);

    /// @brief Basis of the map
    constexpr const TensorBSpline<T, d>& basis() const noexcept
    { return basis_; }

    /**
     * @brief Control points, of size (num_functions,n)
     *
     * @pre The patch outlives any reference taken to them, which a binding
     *      may expose without copying
     */
    constexpr const PointMatrix<T, n>& coefficients() const noexcept
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
     * @param positions Output buffer of size (num_points,n), resized if its
     *        shape changes
     *
     * @pre @p actives and @p values come from the same element. It is not
     *      checked
     */
    void position_on_element(const Eigen::VectorXi& actives,
                             const Eigen::MatrixX<T>& values,
                             PointMatrix<T, n>& positions) const;

private:
    /// @brief Basis of the map
    TensorBSpline<T, d> basis_;

    /// @brief Control points, one row per basis function
    PointMatrix<T, n> coefficients_;
};

} // namespace iguana

#endif // IGUANA_GEOMETRY_PATCH_HPP
