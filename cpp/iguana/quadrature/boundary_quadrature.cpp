/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary_quadrature.hpp"

#include <array>
#include <vector>

#include "iguana/geometry/boundary.hpp"
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

template class BoundaryQuadrature<double, 2>;

} // namespace iguana
