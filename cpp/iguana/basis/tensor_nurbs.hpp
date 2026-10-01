/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_BASIS_TENSOR_NURBS_HPP
#define IGUANA_BASIS_TENSOR_NURBS_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "tensor_bspline.hpp"
#include "iguana/domain/tensor_domain.hpp"

namespace iguana
{

/**
 * @brief Tensor-product NURBS basis, the rational form of a tensor-product
 *        B-spline basis
 *
 * Each function is a B-spline scaled by its weight and divided by the weight
 * function, the sum of all weighted B-splines
 *
 *     R_i = w_i N_i / W,    W = sum_j w_j N_j
 *
 * There is one weight per tensor-product function. The weights do not factor
 * per direction in general, so the basis is not a product of univariate
 * NURBS bases
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class TensorNURBS
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the rational form of a B-spline basis
     *
     * @param bspline B-spline basis that the weights rationalize
     * @param weights One weight per function, in the numbering of the basis
     *
     * @throws std::invalid_argument If there is not one weight per basis
     *         function, or if a weight is not positive
     */
    TensorNURBS(TensorBSpline<T, d> bspline, Eigen::VectorX<T> weights);

    /// @brief B-spline basis that the weights rationalize
    constexpr const TensorBSpline<T, d>& bspline() const noexcept
    { return bspline_; }

    /// @brief Weights, one per basis function
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

    /// @brief Domain, whose elements the basis is defined on
    constexpr const TensorDomain<T, d>& domain() const noexcept
    { return bspline_.domain(); }

    /// @brief Number of basis functions
    constexpr int num_functions() const noexcept
    { return bspline_.num_functions(); }

    /// @brief Number of functions active on each element
    constexpr int num_active() const noexcept
    { return bspline_.num_active(); }

    /**
     * @brief Functions that are non-zero on an element, those of the
     *        B-spline basis
     *
     * @param element Element index
     * @param actives Output vector of num_active() function indices. It is
     *        resized when necessary
     *
     * @pre @p element lies in [0, domain().num_elements())
     */
    void active_on_element(int element, Eigen::VectorXi& actives) const
    { bspline_.active_on_element(element, actives); }

    /**
     * @brief Evaluates the non-zero functions on an element
     *
     * The rows follow the order of active_on_element()
     *
     * @param first_active First active function in each direction
     * @param points Evaluation points, one per row and one coordinate per
     *        column, with size (num_points, dimension)
     * @param values Output matrix of size (num_active(), num_points). It is
     *        resized when necessary
     *
     * @pre @p first_active belongs to an existing element, @p points has
     *      dimension columns, and every point lies inside that element
     */
    void eval_on_element(const std::array<int, d>& first_active,
                         const Eigen::MatrixX<T>& points,
                         Eigen::MatrixX<T>& values) const;

private:
    /// @brief B-spline basis that the weights rationalize
    TensorBSpline<T, d> bspline_;

    /// @brief Weights, one per basis function
    Eigen::VectorX<T> weights_;
};

} // namespace iguana

#endif // IGUANA_BASIS_TENSOR_NURBS_HPP
