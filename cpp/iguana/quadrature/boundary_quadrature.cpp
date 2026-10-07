/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary_quadrature.hpp"

#include <array>
#include <stdexcept>
#include <vector>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/boundary.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/quadrature/box_rule.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
BoundaryQuadrature<T, d>::BoundaryQuadrature(
    const EmbeddedBoundary<T, d>& boundary, const GaussLegendre<T, 1>& rule)
{
    using Face = typename Boundary<T, d>::Face;
    using Piece = typename EmbeddedBoundary<T, d>::Piece;

    // Faces in the parameter space of the grid, which the pieces refer to
    const Boundary<T, d>& pulled_back = boundary.boundary();

    // Elements holding a piece, in increasing order
    std::vector<int> held;

    for (int element = 0; element < boundary.num_elements(); ++element)
        if (!boundary.pieces_on_element(element).empty())
            held.push_back(element);

    // Every piece carries the whole rule
    const int count = rule.num_points();
    const int num_total = boundary.num_pieces() * count;

    elements_ = Eigen::Map<const Eigen::VectorXi>(
        held.data(), static_cast<Eigen::Index>(held.size()));
    offsets_.resize(num_elements() + 1);
    points_.resize(num_total, d);
    weights_.resize(num_total);
    normals_.resize(num_total, d);
    faces_.resize(num_total);

    // Buffers reused over the pieces
    Eigen::MatrixX<T> parameters;
    Eigen::VectorX<T> weights;
    Eigen::MatrixX<T> values;
    std::array<Eigen::MatrixX<T>, 1> gradients;
    Eigen::VectorXi actives;
    PointMatrix<T, d> positions;
    std::array<PointMatrix<T, d>, 1> tangents;
    Eigen::VectorX<T> measures;
    PointMatrix<T, d> normals;

    // First point of the next piece
    int first = 0;
    offsets_(0) = 0;

    for (int position = 0; position < num_elements(); ++position) {
        for (const Piece& piece :
             boundary.pieces_on_element(elements_(position))) {
            const Face& face = pulled_back.face(piece.face);
            const std::array<int, 1> first_active{
                face.basis().bspline().axis(0).first_active(
                    piece.face_element)};

            // The rule placed on the interval of the face parameter
            parameters = rule.points();
            weights = rule.weights();
            BoxRule<T, 1>::map_to_parameter_space({piece.start}, {piece.end},
                                                  parameters, weights);

            face.basis().grad_on_element(first_active, parameters, values,
                                         gradients);
            face.basis().active_on_element(piece.face_element, actives);

            // The face carries the points into the parameter space of the
            // grid, its length element |c'(s)| scaling their weights
            face.position_on_element(actives, values, positions);
            face.tangent_on_element(actives, gradients, tangents);
            Face::measure_on_element(tangents, measures);
            pulled_back.normal_on_element(piece.face, actives, gradients,
                                          normals);

            points_.middleRows(first, count) = positions;
            weights_.segment(first, count) = weights.cwiseProduct(measures);
            normals_.middleRows(first, count) = normals;
            faces_.segment(first, count).setConstant(piece.face);
            first += count;
        }

        offsets_(position + 1) = first;
    }
}

template<std::floating_point T, std::size_t d>
BoundaryQuadrature<T, d>::BoundaryQuadrature(
    const SurrogateBoundary<T, d>& boundary, const GaussLegendre<T, 1>& rule)
{
    using Face = typename SurrogateBoundary<T, d>::Face;

    // Elements holding a face, in increasing order
    std::vector<int> held;

    for (int element = 0; element < boundary.num_elements(); ++element)
        if (!boundary.faces_on_element(element).empty())
            held.push_back(element);

    // Every face carries the whole rule
    const int count = rule.num_points();
    const int num_total = boundary.num_faces() * count;

    elements_ = Eigen::Map<const Eigen::VectorXi>(
        held.data(), static_cast<Eigen::Index>(held.size()));
    offsets_.resize(num_elements() + 1);
    points_.resize(num_total, d);
    weights_.resize(num_total);
    normals_.setZero(num_total, d);
    faces_.resize(num_total);

    // Buffers reused over the faces
    Eigen::MatrixX<T> parameters;
    Eigen::VectorX<T> weights;

    // First point of the next face, and its index in the boundary
    int first = 0;
    int index = 0;
    offsets_(0) = 0;

    for (int position = 0; position < num_elements(); ++position) {
        const int element = elements_(position);

        for (const Face& face : boundary.faces_on_element(element)) {
            // A segment, d being two, runs across its direction
            const std::size_t direction =
                static_cast<std::size_t>(face.direction);
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
            normals_.block(first, direction, count, 1)
                .setConstant(static_cast<T>(face.side));
            faces_.segment(first, count).setConstant(index);

            first += count;
            ++index;
        }

        offsets_(position + 1) = first;
    }
}

template<std::floating_point T, std::size_t d>
template<typename Basis>
    requires (Basis::dimension == d)
void BoundaryQuadrature<T, d>::shift(const Patch<Basis, 2>& patch,
                                     const Boundary<T, 2>& boundary,
                                     int order)
{
    if (order < 0)
        throw std::invalid_argument("BoundaryQuadrature: "
                                    "the order must be non-negative");

    const Basis& basis = patch.basis();

    // Physical positions of the points, element by element
    PointMatrix<T, 2> positions(num_points(), 2);
    Eigen::MatrixX<T> element_points;
    Eigen::MatrixX<T> values;
    Eigen::VectorXi actives;
    PointMatrix<T, 2> element_positions;

    for (int position = 0; position < num_elements(); ++position) {
        const int element = elements_(position);
        const int first = offsets_(position);
        const int count = offsets_(position + 1) - first;

        // First active function of each direction, from the element of each
        // direction with the first one running fastest
        std::array<int, d> first_active{};
        int remaining = element;

        for (std::size_t direction = 0; direction < d; ++direction) {
            const KnotVector<T>& knots = basis.grid().knots(direction);
            const int axis_element = remaining % knots.num_elements();

            first_active[direction] =
                knots.element_span(axis_element) - knots.degree();
            remaining /= knots.num_elements();
        }

        element_points = points_.middleRows(first, count);
        basis.eval_on_element(first_active, element_points, values);
        basis.active_on_element(element, actives);
        patch.position_on_element(actives, values, element_positions);

        positions.middleRows(first, count) = element_positions;
    }

    // Closest points on the boundary, pulled back into parameter space
    const BoundaryProjection<T> projection =
        project_points(boundary, positions);
    PointMatrix<T, 2> parameters;
    patch.invert_points(projection.positions, parameters);

    distances_ = parameters - points_;
    order_ = order;
}

template class BoundaryQuadrature<double, 2>;

template void BoundaryQuadrature<double, 2>::shift(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    int);
template void BoundaryQuadrature<double, 2>::shift(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    int);

} // namespace iguana
