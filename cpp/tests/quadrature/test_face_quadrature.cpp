/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;
using iguana::Boundary;
using iguana::CellType;
using iguana::KnotVector;
using iguana::PointMatrix;
using iguana::TensorBSpline;
using iguana::TensorGrid;
using iguana::TensorNURBS;

using Face = Boundary<double, 2>::Face;
using Gauss = iguana::GaussLegendre<double, 1>;
using Quadrature = iguana::FaceQuadrature<double, 2>;
using Region = iguana::Patch<TensorBSpline<double, 2>, 2>;

/// @brief Patch of linear elements between breaks in each direction,
///        mapped onto itself so that its parameters are its coordinates
Region patch_of(const std::vector<double>& first,
                const std::vector<double>& second)
{
    const auto knots = [](const std::vector<double>& breaks) {
        std::vector<double> values{breaks.front()};
        values.insert(values.end(), breaks.begin(), breaks.end());
        values.push_back(breaks.back());

        return KnotVector<double>(1, values);
    };

    const TensorBSpline<double, 2> basis(
        TensorGrid<double, 2>({knots(first), knots(second)}));

    // The control points of linear functions lie at the breaks
    PointMatrix<double, 2> points(basis.num_functions(), 2);

    for (std::size_t row = 0; row < second.size(); ++row)
        for (std::size_t column = 0; column < first.size(); ++column)
            points.row(static_cast<Eigen::Index>(
                column + first.size() * row)) << first[column], second[row];

    return {basis, points};
}

/// @brief Circle of radius two around a center, counterclockwise, as four
///        quadratic arcs joined at double knots
Face circle(const Eigen::RowVector2d& center)
{
    const double corner = std::sqrt(.5);

    Eigen::VectorXd weights(9);
    weights << 1., corner, 1., corner, 1., corner, 1., corner, 1.;

    PointMatrix<double, 2> points(9, 2);
    points << 2., 0.,
              2., 2.,
              0., 2.,
             -2., 2.,
             -2., 0.,
             -2., -2.,
              0., -2.,
              2., -2.,
              2., 0.;
    points.rowwise() += center;

    const TensorBSpline<double, 1> bspline({BSpline<double>(
        2, {0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.})});

    return {TensorNURBS<double, 1>(bspline, weights), points};
}

} // namespace

TEST_CASE("A shifted face quadrature reaches the closest points of a "
          "boundary", "[face_quadrature]")
{
    // The middle element, [-1, .5] x [-1.5, 1] in parameters, is the only
    // cell inside the circle of radius two around (.3, -.2), and the others
    // are cut, both moved by an offset in physical space
    const Eigen::RowVector2d offset(10., 5.);
    const Region parameters = patch_of({-3., -1., .5, 3.},
                                       {-3., -1.5, 1., 3.});
    const Region square(parameters.basis(),
                        parameters.coefficients().rowwise() + offset);
    std::vector<CellType> cell_types(9, CellType::cut);
    cell_types[4] = CellType::inside;

    const iguana::JumpFaces<double, 2> faces(
        square.basis().grid(),
        iguana::CellClassification<double, 2>(cell_types));
    const Quadrature unshifted(faces, Gauss(3));

    const Eigen::RowVector2d center(.3, -.2);
    const Boundary<double, 2> boundary({circle(center + offset)}, {1});

    Quadrature quadrature = unshifted;
    quadrature.shift(square, boundary, 2);

    // Every interior face separates cells of different types or two cut
    // cells, and in parameters each distance reaches the circle along the
    // radius through its point
    REQUIRE(quadrature.num_faces() == 12);

    for (int point = 0; point < quadrature.num_points(); ++point) {
        const Eigen::RowVector2d from = quadrature.points().row(point);
        const Eigen::RowVector2d radial = (from - center).normalized();

        REQUIRE((from + quadrature.distances().row(point)
                 - (center + 2. * radial)).norm() < 1e-14);
    }

    // The points, weights and normals stay those of the faces
    REQUIRE(quadrature.is_shifted());
    REQUIRE(quadrature.order() == 2);
    REQUIRE_FALSE(unshifted.is_shifted());
    REQUIRE(quadrature.points() == unshifted.points());
    REQUIRE(quadrature.weights() == unshifted.weights());
    REQUIRE(quadrature.normals() == unshifted.normals());

    // A negative order, or a map that is not affine, cannot shift
    PointMatrix<double, 2> bent = square.coefficients();
    bent(5, 0) += .1;
    const Region curved(square.basis(), bent);

    REQUIRE_THROWS_AS(Quadrature(unshifted).shift(square, boundary, -1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Quadrature(unshifted).shift(curved, boundary, 2),
                      std::invalid_argument);
}
