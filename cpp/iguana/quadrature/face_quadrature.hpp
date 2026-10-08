/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_FACE_QUADRATURE_HPP
#define IGUANA_QUADRATURE_FACE_QUADRATURE_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/embedding/ghost_faces.hpp"
#include "iguana/quadrature/gauss_legendre/gauss_legendre.hpp"

namespace iguana
{

/**
 * @brief Quadrature over interior faces of a grid, each between two cells,
 *        grouped face by face
 *
 * A term on a face compares the functions of the cells on either side,
 * each evaluated with the polynomial of its own cell, so the quadrature
 * keeps both cells of every face. The points are stored face after face,
 * and an offset marks where the points of each face start
 *
 * The points, weights and normals lie in the parameter space of the grid,
 * where the weights measure length and the normal of a face is the unit
 * vector of its direction, pointing from the cell before it to the cell
 * after it. The geometry map turns them into physical ones by Nanson's
 * formula, as for a BoundaryQuadrature
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions, two, as for a
 *         BoundaryQuadrature, so that the faces are segments
 */
template<std::floating_point T, std::size_t d>
class FaceQuadrature
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Places a rule on every face where a ghost penalty acts
     *
     * The rule maps from [-1, 1] onto the interval each face spans across
     * its direction, so that the weights measure length in parameter space
     *
     * @param faces Faces of a grid where a ghost penalty acts
     * @param rule Rule placed on every face
     */
    FaceQuadrature(const GhostFaces<T, d>& faces,
                   const GaussLegendre<T, 1>& rule);

    /// @brief Number of faces
    constexpr int num_faces() const noexcept
    { return static_cast<int>(directions_.size()); }

    /// @brief Number of points over all faces
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief Direction normal to each face
    constexpr const Eigen::VectorXi& directions() const noexcept
    { return directions_; }

    /// @brief Cells on either side of each face, with size (num_faces, 2):
    ///        the cell before it along its direction, then the cell after
    ///        it, in the numbering of the grid
    constexpr const Eigen::MatrixX2i& cells() const noexcept
    { return cells_; }

    /// @brief First point of each face, followed by the number of points
    constexpr const Eigen::VectorXi& offsets() const noexcept
    { return offsets_; }

    /// @brief Points in parameter space, with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& points() const noexcept
    { return points_; }

    /// @brief Weights, one per point, measuring length in parameter space
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

    /// @brief Unit normals in parameter space, pointing from the cell before
    ///        each face to the cell after it, with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& normals() const noexcept
    { return normals_; }

private:
    /// @brief Direction normal to each face
    Eigen::VectorXi directions_;

    /// @brief Cells before and after each face
    Eigen::MatrixX2i cells_;

    /// @brief First point of each face, followed by the number of points
    Eigen::VectorXi offsets_;

    /// @brief Points in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> points_;

    /// @brief Weights, one per point
    Eigen::VectorX<T> weights_;

    /// @brief Unit normals in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> normals_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_FACE_QUADRATURE_HPP
