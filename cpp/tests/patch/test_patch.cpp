/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::Patch;
using iguana::TensorBSpline;

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
        sum += axis.knots()[function + knot];

    return sum / axis.degree();
}

} // namespace

TEST_CASE("Patch requires one control point per basis function", "[patch]")
{
    using Surface = Patch<double, 2>;

    const TensorBSpline<double, 2> basis({quadratic(), cubic()});
    const int expected = basis.num_functions();

    REQUIRE_THROWS_AS(Surface(basis,
                              Eigen::MatrixX3d::Zero(expected - 1, 3)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Surface(basis,
                              Eigen::MatrixX3d::Zero(expected + 1, 3)),
                      std::invalid_argument);
    REQUIRE_NOTHROW(Surface(basis, Eigen::MatrixX3d::Zero(expected, 3)));
}

TEST_CASE("Patch reproduces affine maps exactly", "[patch]")
{
    using Basis = std::variant<TensorBSpline<double, 1>,
                               TensorBSpline<double, 2>,
                               TensorBSpline<double, 3>>;

    const Eigen::Vector3d shift(1.5, -.25, 3.);

    Eigen::MatrixXd curve(3, 1);
    curve << 2., -1., .5;

    Eigen::MatrixXd plane(3, 2);
    plane << 2., .5,
            -1., 3.,
             .25, -2.;

    Eigen::MatrixXd volume(3, 3);
    volume << 2., .5, -1.,
             -1., 3., .75,
              .25, -2., 1.5;

    // The zero map leaves the constant map, which holds by partition of
    // unity alone
    const std::array<std::pair<Basis, Eigen::MatrixXd>, 4> cases{{
        {TensorBSpline<double, 2>({quadratic(), cubic()}),
         Eigen::MatrixXd::Zero(3, 2)},
        {TensorBSpline<double, 1>({cubic()}), curve},
        {TensorBSpline<double, 2>({quadratic(), cubic()}), plane},
        {TensorBSpline<double, 3>({quadratic(), cubic(), repeated()}),
         volume}
    }};

    for (const std::pair<Basis, Eigen::MatrixXd>& entry : cases) {
        const Eigen::MatrixXd& map = entry.second;

        std::visit([&map, &shift](const auto& basis) {
            constexpr std::size_t d =
                std::decay_t<decltype(basis)>::dimension;

            INFO("dimension " << d);

            // Sampling an affine function at the Greville abscissae gives
            // control points that reproduce it exactly
            Eigen::MatrixX3d coefficients(basis.num_functions(), 3);
            Eigen::VectorXd node(d);

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

            const Patch<double, d> patch(basis, coefficients);
            Eigen::MatrixXd values;
            Eigen::VectorXi actives;
            Eigen::MatrixX3d positions;

            for (int element = 0; element < basis.num_elements();
                 ++element) {
                INFO("element " << element);
                std::array<int, d> first{};
                Eigen::MatrixXd points(2, d);
                int rest = element;

                for (std::size_t dir = 0; dir < d; ++dir) {
                    const BSpline<double>& axis = basis.axis(dir);
                    const int index = rest % axis.num_elements();
                    const double start = axis.element_start(index);
                    const double width = axis.element_end(index) - start;
                    const double shear = .1 * static_cast<double>(dir);

                    rest /= axis.num_elements();
                    first[dir] = axis.first_active(index);

                    // Shear the fractions so that swapping a point index
                    // for a coordinate index would move the points
                    points(0, dir) = start + (.25 + shear) * width;
                    points(1, dir) = start + (.75 - shear) * width;
                }

                basis.active_on_element(element, actives);
                basis.eval_on_element(first, points, values);
                patch.position_on_element(actives, values, positions);

                REQUIRE(positions.rows() == points.rows());

                for (Eigen::Index pt = 0; pt < points.rows(); ++pt) {
                    INFO("point " << pt);
                    const Eigen::Vector3d exact =
                        map * points.row(pt).transpose() + shift;

                    for (Eigen::Index c = 0; c < 3; ++c)
                        REQUIRE_THAT(positions(pt, c),
                                     WithinAbs(exact[c], 1e-13));
                }
            }
        }, entry.first);
    }
}
