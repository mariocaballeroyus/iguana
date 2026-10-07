/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_BOUNDARY_QUADRATURE_HPP
#define IGUANA_QUADRATURE_BOUNDARY_QUADRATURE_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/embedding/embedded_boundary.hpp"
#include "iguana/embedding/surrogate_boundary.hpp"
#include "iguana/geometry/boundary.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/quadrature/gauss_legendre/gauss_legendre.hpp"

namespace iguana
{

/**
 * @brief Quadrature over a boundary embedded in a grid, grouped by the
 *        elements of the grid
 *
 * Each piece of the boundary carries a rule of its own on its interval of
 * the face parameter, whose points the face maps into the parameter space
 * of the grid. The points are stored element after element, following the
 * pieces within one, and an offset marks where the points of each element
 * start. Only the elements holding a piece of the boundary are held. On a
 * surrogate boundary, the pieces are whole faces of the inside cells
 *
 * The points, weights and normals lie in the parameter space of the grid,
 * where the weights measure length. The geometry map, with Jacobian J,
 * turns a weight w and a normal m into physical ones by Nanson's formula:
 * the vector |det J| J^-T m w has the physical weight as its length and the
 * physical normal as its direction
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions, two, as only curves are
 *         embedded so far, which EmbeddedBoundary asserts, and the faces of
 *         a surrogate boundary are then segments
 */
template<std::floating_point T, std::size_t d>
class BoundaryQuadrature
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Places a rule on every piece of an embedded boundary
     *
     * The rule maps from [-1, 1] onto the interval of the face parameter of
     * each piece, and the face carries its points into the parameter space
     * of the grid, scaling each weight by its length element there
     *
     * @param boundary Boundary divided over the elements of a grid
     * @param rule Rule placed on every piece
     */
    BoundaryQuadrature(const EmbeddedBoundary<T, d>& boundary,
                       const GaussLegendre<T, 1>& rule);

    /**
     * @brief Places a rule on every face of a surrogate boundary
     *
     * The rule maps from [-1, 1] onto the interval each face spans across
     * its direction, so that the weights measure length in parameter space,
     * and the normal of a face is its side times the unit vector of its
     * direction
     *
     * @param boundary Surrogate boundary of the inside cells of a grid
     * @param rule Rule placed on every face
     */
    BoundaryQuadrature(const SurrogateBoundary<T, d>& boundary,
                       const GaussLegendre<T, 1>& rule);

    /**
     * @brief Shifts the quadrature onto a boundary, keeping the distance
     *        from each point to its closest point there and the order of the
     *        Taylor series that cross it
     *
     * This is the shift of the shifted boundary method. The points stay
     * where they are, and so do the weights, normals and faces, so that an
     * integrand is still integrated over this boundary, the surrogate one,
     * with the functions expanded from each point along its distance. The
     * closest points are found in physical space and pulled back through the
     * map of the patch
     *
     * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
     *
     * @param patch Patch whose grid holds the points
     * @param boundary Boundary of curves in the physical space of the patch,
     *        the true one
     * @param order Highest total order of the Taylor series, as in
     *        ElementValues::shift()
     *
     * @throws std::invalid_argument If the order is negative, if the boundary
     *         has nothing to project onto, as for project_points(), or if the
     *         map of the patch is not affine
     *
     * @pre The points lie on the grid of @p patch
     */
    template<typename Basis>
        requires (Basis::dimension == d)
    void shift(const Patch<Basis, 2>& patch, const Boundary<T, 2>& boundary,
               int order);

    /// @brief Number of elements holding a piece of the boundary
    constexpr int num_elements() const noexcept
    { return static_cast<int>(elements_.size()); }

    /// @brief Number of points over all elements
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief Index of each element holding a piece of the boundary, in
    ///        increasing order, in the grid numbering
    constexpr const Eigen::VectorXi& elements() const noexcept
    { return elements_; }

    /// @brief First point of each element, followed by the number of points
    constexpr const Eigen::VectorXi& offsets() const noexcept
    { return offsets_; }

    /// @brief Points in parameter space, with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& points() const noexcept
    { return points_; }

    /// @brief Weights, one per point, measuring length in parameter space
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

    /// @brief Unit normals in parameter space, pointing out of the domain,
    ///        with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& normals() const noexcept
    { return normals_; }

    /// @brief Face of the boundary holding each point, by its index in the
    ///        embedded boundary, or in the surrogate one, counted element
    ///        after element
    constexpr const Eigen::VectorXi& faces() const noexcept
    { return faces_; }

    /// @brief Whether the quadrature is shifted onto a boundary
    constexpr bool is_shifted() const noexcept
    { return distances_.rows() > 0; }

    /// @brief Distance from each point to its closest point on the boundary
    ///        the quadrature is shifted onto, in parameter space, with size
    ///        (num_points, d), empty unless shifted
    constexpr const Eigen::MatrixX<T>& distances() const noexcept
    { return distances_; }

    /// @brief Highest total order of the Taylor series of a shifted
    ///        quadrature, zero unless shifted
    constexpr int order() const noexcept
    { return order_; }

private:
    /// @brief Index of each element holding a piece of the boundary
    Eigen::VectorXi elements_;

    /// @brief First point of each element, followed by the number of points
    Eigen::VectorXi offsets_;

    /// @brief Points in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> points_;

    /// @brief Weights, one per point
    Eigen::VectorX<T> weights_;

    /// @brief Unit normals in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> normals_;

    /// @brief Face of the boundary holding each point
    Eigen::VectorXi faces_;

    /// @brief Distance from each point to its closest point, empty unless
    ///        shifted
    Eigen::MatrixX<T> distances_;

    /// @brief Highest total order of the Taylor series
    int order_ = 0;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_BOUNDARY_QUADRATURE_HPP
