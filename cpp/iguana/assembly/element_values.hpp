/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ASSEMBLY_ELEMENT_VALUES_HPP
#define IGUANA_ASSEMBLY_ELEMENT_VALUES_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Optional values of ElementValues, computed only when flagged
 *
 * The values, parametric gradients, tangents and measures are always
 * computed, as every integral needs the measure and the measure needs the
 * rest
 */
struct ValueFlags
{
    /// @brief Whether to compute the gradients in physical space, J^-T times
    ///        the parametric ones, which only a domain patch has
    bool physical_gradients = false;
};

/**
 * @brief Values of the functions of a patch and of its map at points of one
 *        element at a time
 *
 * The functions are those of the basis of the patch, which also maps the
 * points, so that one evaluation of the basis gives both the functions and
 * the Jacobian J of the map. reinit() moves the values to an element and
 * fills them there, reusing the buffers of the previous element. The rows
 * follow active_on_element(), so that the k-th row pairs with the k-th
 * degree of freedom a space lists on the element
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class ElementValues
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

    /**
     * @brief Constructs the values of the functions of a patch, filled by
     *        reinit()
     *
     * @param patch Patch whose functions are evaluated and which maps the
     *        points
     * @param flags Optional values to compute
     *
     * @throws std::invalid_argument If physical gradients are flagged on a
     *         patch with fewer directions than its space, such as a shell
     *
     * @pre @p patch outlives the values, which keep a reference to it
     */
    ElementValues(const Patch<Basis, n>& patch, ValueFlags flags);

    /**
     * @brief Moves the values to an element and fills them at points of it
     *
     * The buffers are resized only when the number of points changes
     *
     * @param element Element index, in the numbering of the grid of the
     *        basis
     * @param points Points in parameter space, with size (num_points, dim)
     *
     * @pre @p element lies in the grid and every point lies inside it. With
     *      physical gradients flagged, J is also invertible at the points
     */
    void reinit(int element, const Eigen::MatrixX<Scalar>& points);

    /**
     * @brief Replaces the values by their Taylor series from the points of
     *        the last reinit() along shifts, up to a total order
     *
     * This is how the shifted boundary method expands the functions of a
     * surrogate point towards the true boundary. The gradients, tangents,
     * measures and physical gradients stay at the points, where the
     * surrogate boundary is measured
     *
     * @param shifts Shift of each point in parameter space, with size
     *        (num_points, dim)
     * @param order Highest total order of the derivatives kept
     *
     * @pre reinit() was called, @p shifts has one row per point of it, and
     *      @p order is non-negative
     */
    void shift(const Eigen::MatrixX<Scalar>& shifts, int order)
        requires std::same_as<Basis, TensorBSpline<Scalar, dim>>;

    /**
     * @brief Replaces the values by their derivatives across the knot lines
     *        of a direction, of the order of the degree there, in physical
     *        space
     *
     * Across a knot line of maximal continuity, these are the only
     * derivatives that jump, as a ghost penalty compares them. The
     * derivative of order p along the direction is scaled by |J^-T e_k|^p,
     * the length of the physical gradient of its parameter. On an affine map
     * its jump across the knot line is then that of the physical normal
     * derivative of order p, and with orthogonal parametric directions, as
     * on a rectangle, it is that derivative itself. The gradients, tangents,
     * measures and physical gradients stay those of the last reinit()
     *
     * @param direction Direction across the knot lines
     *
     * @pre reinit() was called, and @p direction lies in [0, dim)
     */
    void differentiate(std::size_t direction)
        requires std::same_as<Basis, TensorBSpline<Scalar, dim>>;

    /// @brief Values of the active functions, of size
    ///        (num_active, num_points)
    constexpr const Eigen::MatrixX<Scalar>& values() const noexcept
    { return values_; }

    /// @brief Derivatives of the active functions along each parametric
    ///        direction, one matrix per direction of the size of values()
    constexpr const std::array<Eigen::MatrixX<Scalar>, dim>& gradients()
        const noexcept
    { return gradients_; }

    /// @brief Derivatives of the map along each parametric direction, the
    ///        columns of J, one buffer of size (num_points, n) per direction
    constexpr const std::array<PointMatrix<Scalar, n>, dim>& tangents()
        const noexcept
    { return tangents_; }

    /**
     * @brief Measure of the patch at each point, the factor that turns a
     *        parametric weight into a physical one
     *
     * It is |det J| on a domain and sqrt(det(J^T J)) on a curve or surface.
     * At points of an embedded boundary it measures the patch, not the
     * boundary, whose weights need the boundary normals as well
     */
    constexpr const Eigen::VectorX<Scalar>& measures() const noexcept
    { return measures_; }

    /**
     * @brief Size h of the element at each point, the dim-th root of its
     *        measure in physical space as the map gives it there
     *
     * It is (measure x volume of the element in parameter space)^(1/dim),
     * the side of a cube of the same measure, as penalties scaled by the
     * element size need it
     */
    constexpr const Eigen::VectorX<Scalar>& sizes() const noexcept
    { return sizes_; }

    /**
     * @brief Gradients of the active functions in physical space, one
     *        matrix per coordinate of the size of values()
     *
     * @pre Physical gradients are flagged. It is not checked
     */
    constexpr const std::array<Eigen::MatrixX<Scalar>, n>&
    physical_gradients() const noexcept requires (dim == n)
    { return physical_gradients_; }

private:
    /// @brief Patch whose functions are evaluated
    const Patch<Basis, n>& patch_;

    /// @brief Optional values to compute
    ValueFlags flags_;

    /// @brief First active function of each direction on the element
    std::array<int, dim> first_active_{};

    /// @brief Points of the element in parameter space
    Eigen::MatrixX<Scalar> points_;

    /// @brief Functions active on the element
    Eigen::VectorXi actives_;

    /// @brief Values of the active functions
    Eigen::MatrixX<Scalar> values_;

    /// @brief Derivatives of the active functions along each direction
    std::array<Eigen::MatrixX<Scalar>, dim> gradients_;

    /// @brief Derivatives of the map along each direction
    std::array<PointMatrix<Scalar, n>, dim> tangents_;

    /// @brief Measure of the patch at each point
    Eigen::VectorX<Scalar> measures_;

    /// @brief Size of the element at each point
    Eigen::VectorX<Scalar> sizes_;

    /// @brief Gradients of the active functions in physical space, empty
    ///        unless flagged
    std::array<Eigen::MatrixX<Scalar>, n> physical_gradients_;
};

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ELEMENT_VALUES_HPP
