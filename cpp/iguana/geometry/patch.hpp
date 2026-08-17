/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_GEOMETRY_PATCH_HPP
#define IGUANA_GEOMETRY_PATCH_HPP

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
     * @brief Maps points of an element into physical space.
     *
     * The physical points are the control points of the active
     * functions weighted by the function values, so that the basis
     * evaluation of the caller is reused rather than repeated.
     *
     * @param actives The functions that are non-zero on the element, as
     *        given by active_on_element().
     * @param values Their values at the points, of size
     *        (num_active,num_points), as given by eval_on_element().
     * @param physical Output buffer of size (num_points,3), resized if
     *        its shape changes.
     *
     * @pre @p actives and @p values come from the same element. It is
     *      not checked.
     */
    void eval(const Eigen::VectorXi& actives,
              const Eigen::MatrixX<T>& values,
              Eigen::MatrixX3<T>& physical) const;

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
