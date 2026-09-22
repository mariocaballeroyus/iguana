/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_PATCH_PATCH_HPP
#define IGUANA_PATCH_PATCH_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"

namespace iguana
{

/**
 * @brief Matrix of points in physical space, one per row. The storage is
 *        row-major, so that a point is contiguous.
 *
 * @tparam T Floating-point type of the coordinates
 */
template<std::floating_point T>
using PointMatrix = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>;

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
    Patch(TensorBSpline<T, d> basis, PointMatrix<T> coefficients);

    /// @brief Basis of the map
    constexpr const TensorBSpline<T, d>& basis() const noexcept
    { return basis_; }

    /**
     * @brief Control points, of size (num_functions,3)
     *
     * @pre The patch outlives any reference taken to them, which a binding
     *      may expose without copying
     */
    constexpr const PointMatrix<T>& coefficients() const noexcept
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
                             PointMatrix<T>& positions) const;

    /**
     * @brief Isocurves of the patch along every knot line
     *
     * Each curve lies in the univariate space of the direction it runs
     * along, so that it keeps its degree and knots and reproduces the
     * patch along its line rather than sampling it
     *
     * @return One group of curves per direction, holding the knot lines of
     *         the remaining directions with the first of them running
     *         fastest. A univariate patch pins nothing, so its single
     *         group holds the patch itself
     */
    std::array<std::vector<Patch<T, 1>>, d> isocurves() const;

private:
    /**
     * @brief Isocurve running along one direction, at knot lines of the
     *        others
     *
     * Only the control points are formed, by contracting the control net
     * against the values of the pinned directions at their knot lines
     *
     * @param direction Direction the curve runs along
     * @param lines Knot line pinning each remaining direction, in
     *        increasing order of direction
     *
     * @return Curve patch of the isocurve
     *
     * @pre @p direction is a direction of the patch and every line lies in
     *      [0,num_elements] of its own axis, which isocurves() ensures
     */
    Patch<T, 1> isocurve(std::size_t direction,
                         const std::array<int, d - 1>& lines) const;

    /// @brief Basis of the map
    TensorBSpline<T, d> basis_;

    /// @brief Control points, one row per basis function
    PointMatrix<T> coefficients_;
};

} // namespace iguana

#endif // IGUANA_PATCH_PATCH_HPP
