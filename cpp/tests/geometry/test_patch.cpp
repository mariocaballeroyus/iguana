/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::Patch;
using iguana::TensorBSpline;
using iguana::TensorNURBS;

using Points = iguana::PointMatrix<double, 3>;

BSpline<double> quadratic()
{
    return BSpline<double>(2, {0., 0., 0., .4, 1., 1., 1.});
}

BSpline<double> cubic()
{
    return BSpline<double>(3, {0., 0., 0., 0., .6, 1., 1., 1., 1.});
}

BSpline<double> repeated()
{
    return BSpline<double>(2, {0., 0., 0., .5, .5, 1., 1., 1.});
}

double greville(const BSpline<double>& axis, int function)
{
    double sum = 0.;

    for (int knot = 1; knot <= axis.degree(); ++knot)
        sum += axis.knots().values()[function + knot];

    return sum / axis.degree();
}

/**
 * @brief Checks that a patch reproduces an affine map and its tangents at
 *        points of every element
 *
 * Sampling an affine map at the Greville abscissae gives control points
 * that reproduce it exactly
 *
 * @param basis Basis of the patch
 * @param map Linear part of the map, from parameters to physical space
 * @param shift Constant part of the map
 */
template<std::size_t d, std::size_t n>
void check_affine(const TensorBSpline<double, d>& basis,
                  const Eigen::Matrix<double, n, d>& map,
                  const Eigen::Vector<double, n>& shift)
{
    INFO("directions " << d << ", dimensions " << n);

    iguana::PointMatrix<double, n> coefficients(basis.num_functions(), n);
    Eigen::Vector<double, d> node;

    for (int fun = 0; fun < basis.num_functions(); ++fun) {
        int rest = fun;

        for (std::size_t dir = 0; dir < d; ++dir) {
            const BSpline<double>& axis = basis.axis(dir);

            node[static_cast<Eigen::Index>(dir)] =
                greville(axis, rest % axis.num_functions());
            rest /= axis.num_functions();
        }

        coefficients.row(fun) = (map * node + shift).transpose();
    }

    const Patch<TensorBSpline<double, d>, n> patch(basis, coefficients);
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    iguana::PointMatrix<double, n> positions;
    std::array<Eigen::MatrixXd, d> gradients;
    std::array<iguana::PointMatrix<double, n>, d> tangents;

    for (int element = 0; element < basis.grid().num_elements();
         ++element) {
        INFO("element " << element);
        std::array<int, d> first{};
        Eigen::MatrixXd points(2, d);
        int rest = element;

        for (std::size_t dir = 0; dir < d; ++dir) {
            const BSpline<double>& axis = basis.axis(dir);
            const iguana::KnotVector<double>& knots = axis.knots();
            const int index = rest % knots.num_elements();
            const double start = knots.element_start(index);
            const double width = knots.element_end(index) - start;
            const double shear = .1 * static_cast<double>(dir);

            rest /= knots.num_elements();
            first[dir] = axis.first_active(index);

            // Shear the fractions so that swapping a point index for a
            // coordinate index would move the points
            points(0, dir) = start + (.25 + shear) * width;
            points(1, dir) = start + (.75 - shear) * width;
        }

        basis.active_on_element(element, actives);
        basis.eval_on_element(first, points, values);
        patch.position_on_element(actives, values, positions);

        REQUIRE(positions.rows() == points.rows());

        for (Eigen::Index pt = 0; pt < points.rows(); ++pt) {
            INFO("point " << pt);
            const Eigen::Vector<double, n> exact =
                map * points.row(pt).transpose() + shift;

            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(n); ++c)
                REQUIRE_THAT(positions(pt, c), WithinAbs(exact[c], 1e-13));
        }

        // The tangents of an affine map are the columns of its linear part
        basis.grad_on_element(first, points, values, gradients);
        patch.tangent_on_element(actives, gradients, tangents);

        for (std::size_t dir = 0; dir < d; ++dir) {
            INFO("direction " << dir);
            const Eigen::Index column = static_cast<Eigen::Index>(dir);

            for (Eigen::Index pt = 0; pt < points.rows(); ++pt)
                for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(n); ++c)
                    REQUIRE_THAT(tangents[dir](pt, c),
                                 WithinAbs(map(c, column), 1e-12));
        }
    }
}

} // namespace

TEST_CASE("Patch requires one control point per basis function", "[patch]")
{
    using Surface = Patch<TensorBSpline<double, 2>, 3>;

    const TensorBSpline<double, 2> basis({quadratic(), cubic()});
    const int expected = basis.num_functions();

    REQUIRE_THROWS_AS(Surface(basis,
                              Points::Zero(expected - 1, 3)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Surface(basis,
                              Points::Zero(expected + 1, 3)),
                      std::invalid_argument);
    REQUIRE_NOTHROW(Surface(basis, Points::Zero(expected, 3)));
}

TEST_CASE("Patch reproduces affine maps exactly", "[patch]")
{
    Eigen::Matrix<double, 3, 1> curve;
    curve << 2., -1., .5;

    Eigen::Matrix<double, 3, 2> surface;
    surface << 2., .5,
              -1., 3.,
               .25, -2.;

    Eigen::Matrix<double, 3, 3> volume;
    volume << 2., .5, -1.,
             -1., 3., .75,
              .25, -2., 1.5;

    const Eigen::Vector3d shift(1.5, -.25, 3.);

    // The zero map leaves the constant map, which holds by partition of
    // unity alone
    check_affine<2, 3>(TensorBSpline<double, 2>({quadratic(), cubic()}),
                       Eigen::Matrix<double, 3, 2>::Zero(), shift);

    // Curves, surfaces and volumes in space
    check_affine<1, 3>(TensorBSpline<double, 1>({cubic()}), curve, shift);
    check_affine<2, 3>(TensorBSpline<double, 2>({quadratic(), cubic()}),
                       surface, shift);
    check_affine<3, 3>(TensorBSpline<double, 3>({quadratic(), cubic(),
                                                 repeated()}),
                       volume, shift);

    // Curves and regions in the plane
    check_affine<1, 2>(TensorBSpline<double, 1>({cubic()}),
                       curve.topRows<2>(), shift.head<2>());
    check_affine<2, 2>(TensorBSpline<double, 2>({quadratic(), cubic()}),
                       surface.topRows<2>(), shift.head<2>());
}

TEST_CASE("NURBS patch reproduces a half annulus exactly", "[patch]")
{
    // Two quarter circles of unit radius joined at a double knot, swept
    // along a radius running linearly from 1 to 2
    const TensorBSpline<double, 2> bspline(
        {BSpline<double>(2, {0., 0., 0., .5, .5, 1., 1., 1.}),
         BSpline<double>(1, {0., 0., 1., 1.})});

    Eigen::Matrix<double, 5, 2> circle;
    circle << 1., 0.,
              1., 1.,
              0., 1.,
             -1., 1.,
             -1., 0.;

    const double corner = std::sqrt(.5);
    Eigen::Vector<double, 5> circle_weights;
    circle_weights << 1., corner, 1., corner, 1.;

    // The weights vary along the first direction alone, so that a weight
    // gathered along the wrong direction breaks the circle
    Eigen::VectorXd weights(bspline.num_functions());
    iguana::PointMatrix<double, 2> coefficients(bspline.num_functions(), 2);

    for (int ring = 0; ring < 2; ++ring) {
        for (int arc = 0; arc < 5; ++arc) {
            weights(arc + 5 * ring) = circle_weights(arc);
            coefficients.row(arc + 5 * ring) = (1. + ring) * circle.row(arc);
        }
    }

    const Patch<TensorNURBS<double, 2>, 2> patch(
        TensorNURBS<double, 2>(bspline, weights), coefficients);
    const TensorNURBS<double, 2>& basis = patch.basis();

    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    iguana::PointMatrix<double, 2> positions;
    std::array<Eigen::MatrixXd, 2> gradients;
    std::array<iguana::PointMatrix<double, 2>, 2> tangents;

    for (const auto& element : basis.grid()) {
        INFO("element " << element.index());
        Eigen::MatrixXd points(2, 2);

        for (std::size_t dir = 0; dir < 2; ++dir) {
            const double start = element.start()[dir];
            const double width = element.end()[dir] - start;
            const double shear = .1 * static_cast<double>(dir);

            points(0, dir) = start + (.25 + shear) * width;
            points(1, dir) = start + (.75 - shear) * width;
        }

        basis.active_on_element(element.index(), actives);
        basis.grad_on_element(element.first_active(), points, values,
                              gradients);
        patch.position_on_element(actives, values, positions);
        patch.tangent_on_element(actives, gradients, tangents);

        for (Eigen::Index pt = 0; pt < points.rows(); ++pt) {
            INFO("point " << pt);
            const Eigen::Vector2d position = positions.row(pt).transpose();
            const double radius = position.norm();

            // The radius follows the second parameter, so the first moves
            // along a circle and the second along a ray
            REQUIRE_THAT(radius, WithinAbs(1. + points(pt, 1), 1e-14));
            REQUIRE_THAT(position.dot(tangents[0].row(pt).transpose()),
                         WithinAbs(0., 1e-13));

            for (Eigen::Index c = 0; c < 2; ++c)
                REQUIRE_THAT(tangents[1](pt, c),
                             WithinAbs(position[c] / radius, 1e-13));
        }
    }
}
