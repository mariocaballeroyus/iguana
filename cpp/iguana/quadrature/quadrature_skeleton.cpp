/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "quadrature_skeleton.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

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

template class QuadratureSkeleton<double, 2>;

} // namespace iguana
