/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_QUADRATURE_BOUNDARY_HPP
#define IGUANA_QUADRATURE_QUADRATURE_BOUNDARY_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/embedding/embedded_boundary.hpp"
#include "iguana/embedding/surrogate_boundary.hpp"
#include "iguana/quadrature/gauss_legendre/gauss_legendre.hpp"
#include "iguana/quadrature/quadrature.hpp"

namespace iguana
{

/**
 * @brief Quadrature over a boundary embedded in a grid, grouped by the
 *        elements of the grid
 *
 * Each piece of the boundary carries a rule of its own on its interval of
 * the face parameter, whose points the face maps into the parameter space
 * of the grid. Its groups are the elements holding a piece of the
 * boundary, with the points following the pieces within one. On a
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
class QuadratureBoundary : public Quadrature<T, d>
{
public:
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
    QuadratureBoundary(const EmbeddedBoundary<T, d>& boundary,
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
    QuadratureBoundary(const SurrogateBoundary<T, d>& boundary,
                       const GaussLegendre<T, 1>& rule);

    /// @brief Number of elements holding a piece of the boundary
    constexpr int num_elements() const noexcept
    { return static_cast<int>(elements_.size()); }

    /// @brief Index of each element holding a piece of the boundary, in
    ///        increasing order, in the grid numbering
    constexpr const Eigen::VectorXi& elements() const noexcept
    { return elements_; }

    /// @brief Unit normals in parameter space, pointing out of the domain,
    ///        with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& normals() const noexcept
    { return normals_; }

    /// @brief Face of the boundary holding each point, by its index in the
    ///        embedded boundary, or in the surrogate one, counted element
    ///        after element
    constexpr const Eigen::VectorXi& faces() const noexcept
    { return faces_; }

private:
    using Quadrature<T, d>::offsets_;
    using Quadrature<T, d>::points_;
    using Quadrature<T, d>::weights_;

    /// @brief Index of each element holding a piece of the boundary
    Eigen::VectorXi elements_;

    /// @brief Unit normals in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> normals_;

    /// @brief Face of the boundary holding each point
    Eigen::VectorXi faces_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_QUADRATURE_BOUNDARY_HPP
