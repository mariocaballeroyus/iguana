/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "face_quadrature.hpp"

#include <cstddef>

#include "iguana/quadrature/box_rule.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
FaceQuadrature<T, d>::FaceQuadrature(const GhostFaces<T, d>& faces,
                                     const GaussLegendre<T, 1>& rule)
{
    using Face = typename GhostFaces<T, d>::Face;

    // Every face carries the whole rule
    const int count = rule.num_points();
    const int num_total = faces.num_faces() * count;

    directions_.resize(faces.num_faces());
    cells_.resize(faces.num_faces(), 2);
    offsets_.resize(faces.num_faces() + 1);
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

    for (const Face& face : faces.faces()) {
        // A segment, d being two, runs across its direction
        const std::size_t direction = static_cast<std::size_t>(face.direction);
        const std::size_t across = 1 - direction;

        // The rule placed on the interval the face spans
        parameters = rule.points();
        weights = rule.weights();
        BoxRule<T, 1>::map_to_parameter_space({face.start[across]},
                                              {face.end[across]},
                                              parameters, weights);

        points_.block(first, across, count, 1) = parameters;
        points_.block(first, direction, count, 1)
            .setConstant(face.start[direction]);
        weights_.segment(first, count) = weights;
        normals_.block(first, direction, count, 1).setOnes();

        directions_(index) = face.direction;
        cells_(index, 0) = face.before;
        cells_(index, 1) = face.after;

        first += count;
        ++index;
        offsets_(index) = first;
    }
}

template class FaceQuadrature<double, 2>;

} // namespace iguana
