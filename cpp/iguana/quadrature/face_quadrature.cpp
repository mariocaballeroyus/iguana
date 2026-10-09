/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "face_quadrature.hpp"

#include <cstddef>
#include <stdexcept>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/quadrature/box_rule.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
FaceQuadrature<T, d>::FaceQuadrature(const GhostFaces<T, d>& faces,
                                     const GaussLegendre<T, 1>& rule)
{
    place(faces, rule);
}

template<std::floating_point T, std::size_t d>
FaceQuadrature<T, d>::FaceQuadrature(const JumpFaces<T, d>& faces,
                                     const GaussLegendre<T, 1>& rule)
{
    place(faces, rule);
}

template<std::floating_point T, std::size_t d>
template<typename Faces>
void FaceQuadrature<T, d>::place(const Faces& faces,
                                 const GaussLegendre<T, 1>& rule)
{
    using Face = typename Faces::Face;

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

template<std::floating_point T, std::size_t d>
template<typename Basis>
    requires (Basis::dimension == d)
void FaceQuadrature<T, d>::shift(const Patch<Basis, 2>& patch,
                                 const Boundary<T, 2>& boundary, int order)
{
    if (order < 0)
        throw std::invalid_argument("FaceQuadrature: "
                                    "the order must be non-negative");

    if (!patch.is_affine())
        throw std::invalid_argument("FaceQuadrature: "
                                    "the map of the patch must be affine");

    // Physical positions of the points through the map, x = a + A xi
    PointMatrix<T, 2> positions = points_ * patch.linear_part().transpose();
    positions.rowwise() += patch.offset().transpose();

    // Closest points on the boundary, pulled back into parameter space
    const BoundaryProjection<T> projection =
        project_points(boundary, positions);
    PointMatrix<T, 2> parameters;
    patch.invert_points(projection.positions, parameters);

    distances_ = parameters - points_;
    order_ = order;
}

template class FaceQuadrature<double, 2>;

template void FaceQuadrature<double, 2>::shift(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    int);
template void FaceQuadrature<double, 2>::shift(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    int);

} // namespace iguana
