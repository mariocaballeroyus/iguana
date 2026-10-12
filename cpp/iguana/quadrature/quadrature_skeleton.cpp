/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "quadrature_skeleton.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/quadrature/box_rule.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
QuadratureSkeleton<T, d>::QuadratureSkeleton(
    const TensorGrid<T, d>& grid,
    std::span<const int> faces,
    const GaussLegendre<T, 1>& rule)
{
    for (const int face : faces) {
        if (face < 0 || face >= grid.num_faces())
            throw std::invalid_argument("QuadratureSkeleton: "
                                        "the faces must be interior faces "
                                        "of the grid");
    }

    // Every face carries the whole rule
    const int count = rule.num_points();
    const int num_total = static_cast<int>(faces.size()) * count;

    directions_.resize(faces.size());
    cells_.resize(faces.size(), 2);
    offsets_.resize(faces.size() + 1);
    points_.resize(num_total, d);
    weights_.resize(num_total);
    normals_.setZero(num_total, d);

    // Buffers reused over the faces
    Eigen::MatrixX<T> parameters;
    Eigen::VectorX<T> weights;

    // First point of the next face, and its index
    int first = 0;
    int index = 0;
    offsets_(0) = 0;

    for (const int face : faces) {
        // A segment, d being two, runs across its direction
        const std::size_t direction = grid.face_direction(face);
        const std::size_t across = 1 - direction;
        const std::array<T, d> start = grid.face_start(face);
        const std::array<T, d> end = grid.face_end(face);
        const std::array<int, 2> cells = grid.face_elements(face);

        // The rule placed on the interval the face spans
        parameters = rule.points();
        weights = rule.weights();

        BoxRule<T, 1>::map_to_parameter_space(
            {start[across]}, {end[across]}, parameters, weights);

        points_.block(first, across, count, 1) = parameters;
        points_.block(first, direction, count, 1).setConstant(start[direction]);
        weights_.segment(first, count) = weights;
        normals_.block(first, direction, count, 1).setOnes();

        directions_(index) = static_cast<int>(direction);
        cells_.row(index) << cells[0], cells[1];

        first += count;
        ++index;
        offsets_(index) = first;
    }
}

template<std::floating_point T, std::size_t d>
template<typename Basis> requires (Basis::dimension == d)
void QuadratureSkeleton<T, d>::project_to_closest(
    const Patch<Basis, 2>& patch,
    const Boundary<T, 2>& boundary,
    int order)
{
    if (order < 0)
        throw std::invalid_argument("QuadratureSkeleton: "
                                    "the order must be non-negative");

    if (!patch.is_affine())
        throw std::invalid_argument("QuadratureSkeleton: "
                                    "the map of the patch must be affine");

    // Physical positions of the points through the map, x = a + A xi
    PointMatrix<T, 2> positions = points_ * patch.linear_part().transpose();
    positions.rowwise() += patch.offset().transpose();

    // Closest points on the boundary in the physical space
    const BoundaryProjection<T> projection
        = project_points(boundary, positions);

    // Pulled back into parameter space
    PointMatrix<T, 2> parameters;
    patch.invert_points(projection.positions, parameters);

    distances_ = parameters - points_;
    order_ = order;
}

template class QuadratureSkeleton<double, 2>;

template void QuadratureSkeleton<double, 2>::project_to_closest(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    int);
template void QuadratureSkeleton<double, 2>::project_to_closest(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    int);

} // namespace iguana
