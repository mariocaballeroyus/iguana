/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_PATCH_HPP
#define IGUANA_GEOMETRY_PATCH_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"

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
 * @brief Tensor-product B-spline or NURBS patch, a map from the parameter
 *        box into physical space
 *
 * The patch pairs a basis with one control point per basis function, so that
 * a parameter maps to physical space. A physical space of more dimensions
 * than parametric directions makes the patch a curve or surface in it, such
 * as a shell, and one of as many a domain, such as a planar region
 *
 * @tparam Basis Tensor-product basis of the map, TensorBSpline or
 *         TensorNURBS
 * @tparam n Dimension of the physical space, at least the number of
 *         parametric directions
 */
template<typename Basis, std::size_t n>
class Patch
{
    static_assert(Basis::dimension <= n,
                  "Patch: "
                  "the physical space needs a dimension per parametric "
                  "direction");

public:
    /// @brief Floating-point type of the control points
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

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
    Patch(Basis basis, PointMatrix<Scalar, n> coefficients);

    /// @brief Basis of the map
    constexpr const Basis& basis() const noexcept
    { return basis_; }

    /**
     * @brief Control points, of size (num_functions,n)
     *
     * @pre The patch outlives any reference taken to them, which a binding
     *      may expose without copying
     */
    constexpr const PointMatrix<Scalar, n>& coefficients() const noexcept
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
                             const Eigen::MatrixX<Scalar>& values,
                             PointMatrix<Scalar, n>& positions) const;

    /**
     * @brief Tangents of the patch at points of an element, the derivatives
     *        of the map along each parametric direction
     *
     * They are the columns of the Jacobian of the map
     *
     * @param actives Functions that are non-zero on the element, as given by
     *        active_on_element()
     * @param gradients Their derivatives along each direction at the points,
     *        each of size (num_active,num_points), as given by
     *        grad_on_element()
     * @param tangents Output with one buffer per direction, each of size
     *        (num_points,n), resized if its shape changes
     *
     * @pre @p actives and @p gradients come from the same element. It is not
     *      checked
     */
    void tangent_on_element(
        const Eigen::VectorXi& actives,
        const std::array<Eigen::MatrixX<Scalar>, dim>& gradients,
        std::array<PointMatrix<Scalar, n>, dim>& tangents) const;

    /**
     * @brief Measure of the patch at points of an element, the factor that
     *        turns a parametric measure into a physical one
     *
     * It is |det J| when the patch has as many directions as its space, and
     * sqrt(det(J^T J)) otherwise: the length of the tangent along a curve
     * and the area its tangents span on a surface
     *
     * @param tangents Tangents at the points, one buffer of size
     *        (num_points,n) per direction, as given by tangent_on_element()
     * @param measures Output vector of size num_points, resized if its size
     *        changes
     */
    static void measure_on_element(
        const std::array<PointMatrix<Scalar, n>, dim>& tangents,
        Eigen::VectorX<Scalar>& measures);

    /**
     * @brief Gradients in physical space of the active functions at points
     *        of an element, from their parametric gradients
     *
     * By the chain rule, the physical gradient is J^-T times the parametric
     * one, so that the gradients of all active functions at a point share
     * one inverse
     *
     * @param tangents Tangents at the points, one buffer of size
     *        (num_points,n) per direction, as given by tangent_on_element()
     * @param gradients Derivatives of the active functions along each
     *        direction, each of size (num_active,num_points), as given by
     *        grad_on_element()
     * @param physical_gradients Output with one matrix per coordinate of the
     *        space, each of the size of the gradients, resized when necessary
     *
     * @pre J is invertible at every point
     */
    static void physical_grad_on_element(
        const std::array<PointMatrix<Scalar, n>, dim>& tangents,
        const std::array<Eigen::MatrixX<Scalar>, dim>& gradients,
        std::array<Eigen::MatrixX<Scalar>, n>& physical_gradients)
        requires (dim == n);

    /**
     * @brief Whether the map is affine, x = a + A xi, decided on
     *        construction
     *
     * B-splines reproduce the identity with their Greville abscissae, so the
     * map is affine exactly when each control point is the affine image of
     * its Greville point and, for NURBS, the weights are equal
     */
    constexpr bool is_affine() const noexcept
    { return affine_; }

private:
    /// @brief Basis of the map
    Basis basis_;

    /// @brief Control points, one row per basis function
    PointMatrix<Scalar, n> coefficients_;

    /// @brief Whether the map is affine
    bool affine_;

    /// @brief Offset a of the map if it is affine, zero otherwise
    Eigen::Vector<Scalar, n> offset_;

    /// @brief Linear part A of the map if it is affine, one column per
    ///        direction, zero otherwise
    Eigen::Matrix<Scalar, n, dim> linear_;
};

} // namespace iguana

#endif // IGUANA_GEOMETRY_PATCH_HPP
