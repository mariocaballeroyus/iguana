/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "embedded_boundary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "iguana/embedding/nurbs/grid_crossings.hpp"
#include "iguana/grid/knot_vector.hpp"

namespace iguana
{

namespace
{

/// @brief A coordinate set onto the nearest of some lines when it lies
///        within a tolerance of it, and left as it is otherwise
template<std::floating_point T>
T snapped(T coordinate, const std::vector<T>& lines, T tolerance)
{
    const auto nearest =
        std::ranges::min_element(lines, {}, [coordinate](T line) {
            return std::abs(line - coordinate);
        });

    return std::abs(*nearest - coordinate) <= tolerance ? *nearest
                                                        : coordinate;
}

/**
 * @brief Boundary pulled back into the parameter space of a patch
 *
 * The control points of each face map through the inverse of the patch,
 * exactly for an affine map, and then onto the knot lines they lie within
 * rounding of. A map that reverses orientation turns the normal of the
 * parametrization to the other side, so it flips the sign of each face
 *
 * @throws std::invalid_argument If the map of the patch is not affine
 */
template<typename Basis, std::size_t n>
Boundary<typename Basis::Scalar, n> pull_back(
    const Boundary<typename Basis::Scalar, n>& boundary,
    const Patch<Basis, n>& patch)
{
    using T = typename Basis::Scalar;

    if (!patch.is_affine())
        throw std::invalid_argument("EmbeddedBoundary: "
                                    "the map of the patch must be affine");

    const std::array<std::vector<T>, n> lines = patch.basis().grid().lines();
    const T rounding = 4096 * std::numeric_limits<T>::epsilon();
    const int orientation =
        patch.linear_part().determinant() < T{0} ? -1 : 1;

    std::vector<typename Boundary<T, n>::Face> faces;
    std::vector<int> signs;
    PointMatrix<T, n> parameters;

    for (int face = 0; face < boundary.num_faces(); ++face) {
        const typename Boundary<T, n>::Face& given = boundary.face(face);
        patch.invert_points(given.coefficients(), parameters);

        for (std::size_t axis = 0; axis < n; ++axis) {
            const T tolerance =
                rounding * (lines[axis].back() - lines[axis].front());

            for (Eigen::Index point = 0; point < parameters.rows(); ++point)
                parameters(point, axis) = snapped(parameters(point, axis),
                                                  lines[axis], tolerance);
        }

        faces.emplace_back(given.basis(), parameters);
        signs.push_back(orientation * boundary.sign(face));
    }

    return {std::move(faces), std::move(signs)};
}

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
template<typename Basis>
    requires (Basis::dimension == d)
EmbeddedBoundary<T, d>::EmbeddedBoundary(const Patch<Basis, d>& patch,
                                         const Boundary<T, d>& boundary)
    : boundary_(pull_back(boundary, patch))
{
    const TensorGrid<T, d>& grid = patch.basis().grid();
    const std::array<std::vector<T>, d> lines = grid.lines();
    const int num_elements = grid.num_elements();

    // Pieces of every face, with the element of the grid holding each
    std::vector<int> elements;
    std::vector<Piece> pieces;

    for (int face = 0; face < boundary_.num_faces(); ++face) {
        const typename Boundary<T, d>::Face& curve = boundary_.face(face);
        const KnotVector<T>& knots = curve.basis().bspline().axis(0).knots();

        // Throws unless the knot vector is clamped
        const std::vector<Eigen::MatrixX<T>> operators =
            extraction_operators(knots);

        for (int face_element = 0; face_element < knots.num_elements();
             ++face_element) {
            const std::vector<CurvePiece<T>> divided = divide_curve(
                curve, face_element, operators[face_element], lines,
                boundary_.sign(face), tolerance);

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

template EmbeddedBoundary<double, 2>::EmbeddedBoundary(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&);
template EmbeddedBoundary<double, 2>::EmbeddedBoundary(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&);

} // namespace iguana
