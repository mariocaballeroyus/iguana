/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::KnotVector;
using iguana::TensorBSpline;
using iguana::TensorDomain;

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

template<std::size_t d>
std::array<int, d> element_of(const TensorBSpline<double, d>& basis,
                              int element)
{
    std::array<int, d> indices{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const int count = basis.axis(direction).knots().num_elements();
        indices[direction] = element % count;
        element /= count;
    }

    return indices;
}

template<std::size_t d>
std::array<int, d> first_active_of(const TensorBSpline<double, d>& basis,
                                   int element)
{
    const std::array<int, d> indices = element_of(basis, element);
    std::array<int, d> first{};

    for (std::size_t direction = 0; direction < d; ++direction)
        first[direction] =
            basis.axis(direction).first_active(indices[direction]);

    return first;
}

template<std::size_t d>
Eigen::MatrixXd points_on(const TensorBSpline<double, d>& basis, int element)
{
    const std::array<int, d> indices = element_of(basis, element);
    const std::array fractions{.2, .5, .8};
    Eigen::MatrixXd points(fractions.size(), d);

    for (std::size_t direction = 0; direction < d; ++direction) {
        const iguana::KnotVector<double>& knots = basis.axis(direction).knots();
        const double start = knots.element_start(indices[direction]);
        const double width = knots.element_end(indices[direction]) - start;

        for (std::size_t point = 0; point < fractions.size(); ++point)
            points(point, direction) =
                start + fractions[(point + direction) % 3] * width;
    }

    return points;
}

double greville(const BSpline<double>& axis, int function)
{
    // Greville abscissae are control coefficients for linear precision
    double sum = 0.;

    for (int knot = 1; knot <= axis.degree(); ++knot)
        sum += axis.knots().values()[function + knot];

    return sum / axis.degree();
}

} // namespace

TEST_CASE("Tensor basis is non-negative and partitions unity on every element",
          "[tensor_bspline]")
{
    using Basis = std::variant<TensorBSpline<double, 1>,
                               TensorBSpline<double, 2>,
                               TensorBSpline<double, 3>>;

    const std::array<Basis, 3> bases{
        TensorBSpline<double, 1>({quadratic()}),
        TensorBSpline<double, 2>({quadratic(), cubic()}),
        TensorBSpline<double, 3>({quadratic(), cubic(), repeated()})
    };
    Eigen::MatrixXd values;

    for (const Basis& entry : bases) {
        std::visit([&values](const auto& basis) {
            INFO("dimension " << basis.dimension);
            const int num_elements = basis.domain().num_elements();

            for (int element = 0; element < num_elements; ++element) {
                INFO("element " << element);
                const Eigen::MatrixXd points = points_on(basis, element);

                basis.eval_on_element(first_active_of(basis, element),
                                      points, values);

                REQUIRE(values.rows() == basis.num_active());
                REQUIRE(values.cols() == points.rows());
                REQUIRE(values.minCoeff() >= 0.);

                for (Eigen::Index point = 0; point < points.rows(); ++point)
                    REQUIRE_THAT(values.col(point).sum(),
                                 WithinAbs(1., 1e-13));
            }
        }, entry);
    }
}

TEST_CASE("Tensor basis reproduces coordinates through active indices",
          "[tensor_bspline]")
{
    const TensorBSpline<double, 3> basis(
        {quadratic(), cubic(), repeated()});
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;

    for (int element = 0; element < basis.domain().num_elements(); ++element) {
        INFO("element " << element);
        const Eigen::MatrixXd points = points_on(basis, element);

        basis.active_on_element(element, actives);
        basis.eval_on_element(first_active_of(basis, element), points, values);

        REQUIRE(actives.size() == values.rows());
        REQUIRE(actives.minCoeff() >= 0);
        REQUIRE(actives.maxCoeff() < basis.num_functions());

        for (Eigen::Index point = 0; point < points.rows(); ++point) {
            INFO("point " << point);
            std::array<double, 3> coordinates{};
            double mixed = 0.;

            // Global active indices select tensor control coefficients
            for (int active = 0; active < actives.size(); ++active) {
                int function = actives[active];
                double coefficient = 1.;

                for (std::size_t direction = 0; direction < 3; ++direction) {
                    const BSpline<double>& axis = basis.axis(direction);
                    const int axis_function = function % axis.num_functions();
                    const double node = greville(axis, axis_function);

                    coordinates[direction] += values(active, point) * node;
                    coefficient *= node;
                    function /= axis.num_functions();
                }

                mixed += values(active, point) * coefficient;
            }

            double expected_mixed = 1.;

            for (std::size_t direction = 0; direction < 3; ++direction) {
                REQUIRE_THAT(coordinates[direction],
                             WithinAbs(points(point, direction), 1e-13));
                expected_mixed *= points(point, direction);
            }

            REQUIRE_THAT(mixed, WithinAbs(expected_mixed, 1e-13));
        }
    }
}

TEST_CASE("A basis is built on the knot vectors of its domain",
          "[tensor_bspline]")
{
    const TensorDomain<double, 3> domain(
        {quadratic().knots(), cubic().knots(), repeated().knots()});
    const TensorBSpline<double, 3> basis(domain);

    // One univariate basis per knot vector, over the two elements of each
    for (std::size_t direction = 0; direction < 3; ++direction)
        REQUIRE(basis.axis(direction).knots().values()
                == domain.knots(direction).values());

    REQUIRE(basis.domain().num_elements() == 2 * 2 * 2);

    // Built from the univariate bases, it has the same domain
    const TensorBSpline<double, 3> from_axes(
        {quadratic(), cubic(), repeated()});

    for (std::size_t direction = 0; direction < 3; ++direction)
        REQUIRE(from_axes.domain().knots(direction).values()
                == domain.knots(direction).values());

    // A degree beyond the univariate bases is rejected, on a valid knot
    // vector clamped on [0, 1]
    const int degree = BSpline<double>::max_degree + 1;
    std::vector<double> knots(degree + 1, 0.);
    knots.resize(2 * (degree + 1), 1.);

    using Univariate = TensorBSpline<double, 1>;
    const TensorDomain<double, 1> beyond(
        std::array{KnotVector<double>(degree, knots)});

    REQUIRE_THROWS_AS(Univariate(beyond), std::invalid_argument);
}
