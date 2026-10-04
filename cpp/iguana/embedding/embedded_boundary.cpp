/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "embedded_boundary.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "iguana/embedding/nurbs/grid_crossings.hpp"
#include "iguana/grid/knot_vector.hpp"

namespace iguana
{

namespace
{

/// @brief Parameter of a curve at a parameter of [0, 1] of one of its
///        elements, the end of the element mapping to its knot exactly
template<std::floating_point T>
T curve_parameter(const KnotVector<T>& knots, int element, T t)
{
    const T start = knots.element_start(element);
    const T end = knots.element_end(element);

    return t == T{1} ? end : start + t * (end - start);
}

} // namespace

template<std::floating_point T, std::size_t d>
EmbeddedBoundary<T, d>::EmbeddedBoundary(const TensorGrid<T, d>& grid,
                                         const Boundary<T, d>& boundary)
{
    const std::array<std::vector<T>, d> lines = grid.lines();
    const int num_elements = grid.num_elements();

    // Pieces of every face, with the element of the grid holding each
    std::vector<int> elements;
    std::vector<Piece> pieces;

    for (int face = 0; face < boundary.num_faces(); ++face) {
        const typename Boundary<T, d>::Face& curve = boundary.face(face);
        const KnotVector<T>& knots = curve.basis().bspline().axis(0).knots();

        // Throws unless the knot vector is clamped
        const std::vector<Eigen::MatrixX<T>> operators =
            extraction_operators(knots);

        for (int face_element = 0; face_element < knots.num_elements();
             ++face_element) {
            const std::vector<CurvePiece<T>> divided = divide_curve(
                curve, face_element, operators[face_element], lines,
                boundary.sign(face), tolerance);

            for (const CurvePiece<T>& piece : divided) {
                const T start =
                    curve_parameter(knots, face_element, piece.start);
                const T end = curve_parameter(knots, face_element, piece.end);

                elements.push_back(piece.cell);
                pieces.push_back({face, face_element, start, end});
            }
        }
    }

    // Count the pieces of each element, then place them element by element,
    // keeping the order of the faces and parameters within one
    offsets_ = Eigen::VectorXi::Zero(num_elements + 1);

    for (const int element : elements)
        ++offsets_(element + 1);

    for (int element = 0; element < num_elements; ++element)
        offsets_(element + 1) += offsets_(element);

    std::vector<int> slots(offsets_.data(), offsets_.data() + num_elements);
    pieces_.resize(pieces.size());

    for (std::size_t entry = 0; entry < pieces.size(); ++entry)
        pieces_[slots[elements[entry]]++] = pieces[entry];
}

template class EmbeddedBoundary<double, 2>;

} // namespace iguana
