/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_QUADRATURE_SKELETON_HPP
#define IGUANA_QUADRATURE_QUADRATURE_SKELETON_HPP

#include <concepts>
#include <cstddef>
#include <span>

#include <Eigen/Core>

#include "iguana/geometry/boundary.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/grid/tensor_grid.hpp"
#include "iguana/quadrature/gauss_legendre/gauss_legendre.hpp"
#include "iguana/quadrature/quadrature.hpp"

namespace iguana
{

/**
 * @brief Quadrature over interior faces of a grid, each between two cells,
 *        grouped face by face
 *
 * @details Used in the Generalized Shifted Boundary Method to integrate the
 *          ghost penalty and the facet conditions, which couple the cells on
 *          either side of each face
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions, two, as for a
 *         QuadratureBoundary, so that the faces are segments
 */
template<std::floating_point T, std::size_t d>
class QuadratureSkeleton : public Quadrature<T, d>
{
public:
    /**
     * @brief Places a rule on every face of a list of interior faces of a
     *        grid
     *
     * @details The rule maps from [-1, 1] onto the interval each face spans
     *          across its direction, so that the weights measure length in
     *          parameter space
     *
     * @param grid Grid whose interior faces are listed
     * @param faces Indices of the faces in the numbering of the grid
     * @param rule Rule placed on every face
     *
     * @throws std::invalid_argument If an index does not number an interior
     *         face of @p grid
     */
    QuadratureSkeleton(const TensorGrid<T, d>& grid,
                       std::span<const int> faces,
                       const GaussLegendre<T, 1>& rule);

    /**
     * @brief Projects each point onto a boundary, keeping the distance vector
     *        to its closest point there and the order of the Taylor series
     *        along it
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
    template<typename Basis> requires (Basis::dimension == d)
    void project_to_closest(const Patch<Basis, 2>& patch,
                            const Boundary<T, 2>& boundary, int order);

    /// @brief Number of faces
    constexpr int num_faces() const noexcept
    { return static_cast<int>(directions_.size()); }

    /// @brief Direction normal to each face
    constexpr const Eigen::VectorXi& directions() const noexcept
    { return directions_; }

    /// @brief Cells on either side of each face, with size (num_faces, 2):
    ///        the cell before it along its direction, then the cell after
    ///        it, in the numbering of the grid
    constexpr const Eigen::MatrixX2i& cells() const noexcept
    { return cells_; }

    /// @brief Unit normals in parameter space, pointing from the cell before
    ///        each face to the cell after it, with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& normals() const noexcept
    { return normals_; }

    /// @brief Whether the quadrature is projected onto a boundary
    constexpr bool is_projected() const noexcept
    { return distances_.rows() > 0; }

    /// @brief Distance vector from each point to its closest point on the
    ///        boundary the quadrature is projected onto, in parameter space,
    ///        with size (num_points, d), empty unless projected
    constexpr const Eigen::MatrixX<T>& distances() const noexcept
    { return distances_; }

    /// @brief Highest total order of the Taylor series of a projected
    ///        quadrature, zero unless projected
    constexpr int order() const noexcept
    { return order_; }

private:
    using Quadrature<T, d>::offsets_;
    using Quadrature<T, d>::points_;
    using Quadrature<T, d>::weights_;

    /// @brief Direction normal to each face
    Eigen::VectorXi directions_;

    /// @brief Cells before and after each face
    Eigen::MatrixX2i cells_;

    /// @brief Unit normals in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> normals_;

    /// @brief Distance vector from each point to its closest point, empty
    ///        unless projected
    Eigen::MatrixX<T> distances_;

    /// @brief Highest total order of the Taylor series
    int order_ = 0;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_QUADRATURE_SKELETON_HPP
