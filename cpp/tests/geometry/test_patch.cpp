/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
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
 * @brief Checks that a patch reproduces an affine map at points of every
 *        element
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

    const Patch<double, d, n> patch(basis, coefficients);
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    iguana::PointMatrix<double, n> positions;

    for (int element = 0; element < basis.domain().num_elements();
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
    }
}

} // namespace

TEST_CASE("Patch requires one control point per basis function", "[patch]")
{
    using Surface = Patch<double, 2, 3>;

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
