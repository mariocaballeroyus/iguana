/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_GEOMETRY_PATCH_HPP
#define IGUANA_GEOMETRY_PATCH_HPP

#include<array>
#include<concepts>

#include "iguana/basis/tensor_product.hpp"

namespace iguana
{

/**
 * @brief Tensor-product B-spline patch, a map from the parameter box
 *        into physical space.
 *
 * The patch pairs a basis with one control point per basis function, so
 * that a parameter maps to the physical space, which is fixed to three 
 * dimensions.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the knots and control points.
 */
template<int d, std::floating_point T>
class Patch
{
public:
    /// @brief The number of parametric directions, \f$ d \f$.
    static constexpr int dimension = d;

public:
    /**
     * @brief Constructs the patch from a basis and its control points.
     *
     * @param basis The tensor-product basis of the map.
     * @param coefficients Control points, of size (num_functions,3), one row 
     *                     per basis function in the numbering of the basis.
     *
     * @throws std::invalid_argument If the rows do not match the number
     *         of basis functions.
     */
    Patch(TensorProductBSpline<d, T> basis, Eigen::MatrixX3<T> coefficients);

    /// @brief The basis of the map.
    constexpr const TensorProductBSpline<d, T>& basis() const noexcept
    { return basis_; }

    /// @brief Control points, of size (num_functions,3).
    constexpr const Eigen::MatrixX3<T>& coefficients() const noexcept
    { return coefficients_; }

    /**
     * @brief The position of points of an element in physical space.
     *
     * The positions are the control points of the active functions
     * weighted by the function values, so that the basis evaluation of
     * the caller is reused rather than repeated: the position vector of
     * the map at every point.
     *
     * @param actives The functions that are non-zero on the element, as
     *        given by active_on_element().
     * @param values Their values at the points, of size
     *        (num_active,num_points), as given by eval_on_element().
     * @param positions Output buffer of size (num_points,3), resized if
     *        its shape changes.
     *
     * @pre @p actives and @p values come from the same element. It is
     *      not checked.
     */
    void position(const Eigen::VectorXi& actives,
                  const Eigen::MatrixX<T>& values,
                  Eigen::MatrixX3<T>& positions) const;

    /**
     * @brief The tangent of every parametric direction at points of an
     *        element.
     *
     * The tangents are the derivatives of the map along the directions,
     * the covariant basis of the patch, and the columns of the Jacobian
     * of the map.
     *
     * @param actives The functions that are non-zero on the element, as
     *        given by active_on_element().
     * @param ders Their first derivatives at the points, of size
     *        (num_active * d,num_points), as eval_ders_on_element()
     *        fills its order-1 buffer.
     * @param tangents One output buffer per direction, each of size
     *        (num_points,3) and resized if its shape changes.
     *
     * @pre @p actives and @p ders come from the same element. It is not
     *      checked.
     */
    void tangents(const Eigen::VectorXi& actives,
                  const Eigen::MatrixX<T>& ders,
                  std::array<Eigen::MatrixX3<T>, d>& tangents) const;

private:
    /// @brief The basis of the map.
    TensorProductBSpline<d, T> basis_;

    /// @brief Control points, one row per basis function.
    Eigen::MatrixX3<T> coefficients_;
};

extern template class Patch<1, double>;
extern template class Patch<2, double>;
extern template class Patch<3, double>;

} // namespace iguana

#endif // IGUANA_GEOMETRY_PATCH_HPP
