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
#include "iguana/embedding/jump_faces.hpp"
#include "iguana/geometry/boundary.hpp"
#include "iguana/geometry/patch.hpp"
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

    /**
     * @brief Places a rule on every face across which the volume fraction
     *        may jump
     *
     * The rule is placed as on the faces of a ghost penalty
     *
     * @param faces Faces of a grid across which the volume fraction may
     *        jump
     * @param rule Rule placed on every face
     */
    FaceQuadrature(const JumpFaces<T, d>& faces,
                   const GaussLegendre<T, 1>& rule);

    /**
     * @brief Shifts the quadrature onto a boundary, keeping the distance
     *        from each point to its closest point there and the order of the
     *        Taylor series that cross it
     *
     * As for a BoundaryQuadrature, the points stay where they are, and so do
     * the weights, normals and cells. Both cells of a face expand their own
     * functions along the same distance. The closest points are found in
     * physical space and pulled back through the map of the patch
     *
     * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
     *
     * @param patch Patch whose grid holds the points
     * @param boundary Boundary of curves in the physical space of the patch,
     *        the true one
     * @param order Highest total order of the Taylor series, as in
     *        ElementValues::shift()
     *
     * @throws std::invalid_argument If the order is negative, if the map of
     *         the patch is not affine, or if the boundary has nothing to
     *         project onto, as for project_points()
     *
     * @pre The points lie on the grid of @p patch
     */
    template<typename Basis>
        requires (Basis::dimension == d)
    void shift(const Patch<Basis, 2>& patch, const Boundary<T, 2>& boundary,
               int order);

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
    /**
     * @brief Places a rule on every face of a list, the work of the
     *        constructors
     *
     * @tparam Faces Faces of a grid, as GhostFaces and JumpFaces list them
     */
    template<typename Faces>
    void place(const Faces& faces, const GaussLegendre<T, 1>& rule);

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

    /// @brief Distance from each point to its closest point, empty unless
    ///        shifted
    Eigen::MatrixX<T> distances_;

    /// @brief Highest total order of the Taylor series
    int order_ = 0;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_FACE_QUADRATURE_HPP
