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
 *          ghost penalty and the skeleton conditions, which couple the cells
 *          on either side of each face
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
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_QUADRATURE_SKELETON_HPP
