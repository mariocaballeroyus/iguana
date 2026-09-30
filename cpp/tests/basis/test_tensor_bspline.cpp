/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
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

/// @brief Basis of one, two and three directions, with mixed degrees and a
///        repeated knot
using Basis = std::variant<TensorBSpline<double, 1>,
                           TensorBSpline<double, 2>,
                           TensorBSpline<double, 3>>;

std::array<Basis, 3> test_bases()
{
    return {TensorBSpline<double, 1>({quadratic()}),
            TensorBSpline<double, 2>({quadratic(), cubic()}),
            TensorBSpline<double, 3>({quadratic(), cubic(), repeated()})};
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
    const std::array<Basis, 3> bases = test_bases();
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

TEST_CASE("Tensor gradients match finite differences of the values",
          "[tensor_bspline]")
{
    const std::array<Basis, 3> bases = test_bases();
    const double step = 1e-6;

    for (const Basis& entry : bases) {
        std::visit([step](const auto& basis) {
            constexpr std::size_t d =
                std::decay_t<decltype(basis)>::dimension;

            INFO("dimension " << d);
            const int num_elements = basis.domain().num_elements();

            Eigen::MatrixXd values;
            Eigen::MatrixXd reference;
            Eigen::MatrixXd above;
            Eigen::MatrixXd below;
            std::array<Eigen::MatrixXd, d> gradients;

            for (int element = 0; element < num_elements; ++element) {
                INFO("element " << element);
                const std::array<int, d> first =
                    first_active_of(basis, element);
                const Eigen::MatrixXd points = points_on(basis, element);

                basis.grad_on_element(first, points, values, gradients);
                basis.eval_on_element(first, points, reference);

                REQUIRE(values.isApprox(reference, 1e-14));

                for (std::size_t direction = 0; direction < d; ++direction) {
                    INFO("direction " << direction);
                    Eigen::MatrixXd shifted = points;

                    shifted.col(direction).array() += step;
                    basis.eval_on_element(first, shifted, above);
                    shifted.col(direction).array() -= 2. * step;
                    basis.eval_on_element(first, shifted, below);

                    const Eigen::MatrixXd difference =
                        (above - below) / (2. * step);
                    const double scale = std::max(
                        1., gradients[direction].cwiseAbs().maxCoeff());

                    REQUIRE((gradients[direction] - difference)
                                .cwiseAbs().maxCoeff() <= 1e-6 * scale);
                }
            }
        }, entry);
    }
}

TEST_CASE("Tensor Hessians match finite differences of the gradients",
          "[tensor_bspline]")
{
    const std::array<Basis, 3> bases = test_bases();
    const double step = 1e-6;

    for (const Basis& entry : bases) {
        std::visit([step](const auto& basis) {
            constexpr std::size_t d =
                std::decay_t<decltype(basis)>::dimension;
            constexpr std::size_t num_pairs = d * (d + 1) / 2;

            INFO("dimension " << d);

            // Directions of each second derivative, pure ones first and
            // mixed ones after in lexicographic order
            std::array<std::array<std::size_t, 2>, num_pairs> pairs{};
            std::size_t pair = 0;

            for (std::size_t direction = 0; direction < d; ++direction)
                pairs[pair++] = {direction, direction};

            for (std::size_t row = 0; row < d; ++row)
                for (std::size_t column = row + 1; column < d; ++column)
                    pairs[pair++] = {row, column};

            Eigen::MatrixXd values;
            Eigen::MatrixXd reference;
            std::array<Eigen::MatrixXd, d> gradients;
            std::array<Eigen::MatrixXd, d> reference_gradients;
            std::array<Eigen::MatrixXd, d> above;
            std::array<Eigen::MatrixXd, d> below;
            std::array<Eigen::MatrixXd, num_pairs> hessians;

            for (int element = 0; element < basis.domain().num_elements();
                 ++element) {
                INFO("element " << element);
                const std::array<int, d> first =
                    first_active_of(basis, element);
                const Eigen::MatrixXd points = points_on(basis, element);

                basis.hess_on_element(first, points, values, gradients,
                                      hessians);
                basis.grad_on_element(first, points, reference,
                                      reference_gradients);

                REQUIRE(values.isApprox(reference, 1e-14));

                for (std::size_t direction = 0; direction < d; ++direction)
                    REQUIRE(gradients[direction].isApprox(
                        reference_gradients[direction], 1e-14));

                for (std::size_t index = 0; index < num_pairs; ++index) {
                    const auto [along, across] = pairs[index];
                    INFO("pair " << along << ", " << across);
                    Eigen::MatrixXd shifted = points;

                    // Differentiate the gradient along one direction of the
                    // pair across the other
                    shifted.col(across).array() += step;
                    basis.grad_on_element(first, shifted, reference, above);
                    shifted.col(across).array() -= 2. * step;
                    basis.grad_on_element(first, shifted, reference, below);

                    const Eigen::MatrixXd difference =
                        (above[along] - below[along]) / (2. * step);
                    const double scale = std::max(
                        1., hessians[index].cwiseAbs().maxCoeff());

                    REQUIRE((hessians[index] - difference)
                                .cwiseAbs().maxCoeff() <= 1e-6 * scale);
                }
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
