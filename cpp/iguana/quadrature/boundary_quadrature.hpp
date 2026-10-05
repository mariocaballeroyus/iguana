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
 * start. Only the elements holding a piece of the boundary are held
 *
 * The points, weights and normals lie in the parameter space of the grid,
 * where the weights measure length. The geometry map, with Jacobian J,
 * turns a weight w and a normal m into physical ones by Nanson's formula:
 * the vector |det J| J^-T m w has the physical weight as its length and the
 * physical normal as its direction
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions, two
 */
template<std::floating_point T, std::size_t d>
class BoundaryQuadrature
{
    static_assert(d == 2, "BoundaryQuadrature: "
                          "the boundary must be made of curves");

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

    /// @brief Face of the boundary holding each point
    constexpr const Eigen::VectorXi& faces() const noexcept
    { return faces_; }

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
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_BOUNDARY_QUADRATURE_HPP
