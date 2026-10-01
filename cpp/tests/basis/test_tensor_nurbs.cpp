/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;
using iguana::TensorBSpline;
using iguana::TensorNURBS;

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

/// @brief Rational form of a B-spline basis, with weights that vary from one
///        function to the next
template<std::size_t d>
TensorNURBS<double, d> uneven(TensorBSpline<double, d> bspline)
{
    Eigen::VectorXd weights(bspline.num_functions());

    for (Eigen::Index function = 0; function < weights.size(); ++function)
        weights(function) = 1. + .6 * std::sin(1.7 * function);

    return TensorNURBS<double, d>(std::move(bspline), std::move(weights));
}

/// @brief Bases of one, two and three directions, with mixed degrees and a
///        repeated knot
using Basis = std::variant<TensorNURBS<double, 1>,
                           TensorNURBS<double, 2>,
                           TensorNURBS<double, 3>>;

std::array<Basis, 3> test_bases()
{
    return {uneven(TensorBSpline<double, 1>({quadratic()})),
            uneven(TensorBSpline<double, 2>({quadratic(), cubic()})),
            uneven(TensorBSpline<double, 3>({quadratic(), cubic(),
                                             repeated()}))};
}

/// @brief Points inside an element, at a different fraction of it in each
///        direction
template<std::size_t d>
Eigen::MatrixXd points_on(
    const iguana::TensorGridIterator<double, d>& element)
{
    const std::array fractions{.2, .5, .8};
    Eigen::MatrixXd points(fractions.size(), d);

    for (std::size_t direction = 0; direction < d; ++direction) {
        const double start = element.start()[direction];
        const double width = element.end()[direction] - start;

        for (std::size_t point = 0; point < fractions.size(); ++point)
            points(point, direction) =
                start + fractions[(point + direction) % 3] * width;
    }

    return points;
}

/// @brief Whether a derivative matches the central difference of the
///        quantity it differentiates, evaluated a step above and below
bool matches_difference(const Eigen::MatrixXd& derivative,
                        const Eigen::MatrixXd& above,
                        const Eigen::MatrixXd& below, double step)
{
    const Eigen::MatrixXd difference = (above - below) / (2. * step);
    const double scale = std::max(1., derivative.cwiseAbs().maxCoeff());

    return (derivative - difference).cwiseAbs().maxCoeff() <= 1e-6 * scale;
}

} // namespace

TEST_CASE("NURBS basis is non-negative and partitions unity",
          "[tensor_nurbs]")
{
    Eigen::MatrixXd values;

    for (const Basis& entry : test_bases()) {
        std::visit([&values](const auto& basis) {
            INFO("dimension " << basis.dimension);

            for (const auto& element : basis.grid()) {
                INFO("element " << element.index());
                const Eigen::MatrixXd points = points_on(element);

                basis.eval_on_element(element.first_active(), points,
                                      values);

                REQUIRE(values.rows() == basis.num_active());
                REQUIRE(values.cols() == points.rows());
                REQUIRE(values.minCoeff() >= 0.);
                REQUIRE((values.colwise().sum().array() - 1.)
                            .abs().maxCoeff() <= 1e-13);
            }
        }, entry);
    }
}

TEST_CASE("NURBS derivatives match finite differences", "[tensor_nurbs]")
{
    const double step = 1e-6;

    for (const Basis& entry : test_bases()) {
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
            Eigen::MatrixXd above;
            Eigen::MatrixXd below;
            std::array<Eigen::MatrixXd, d> gradients;
            std::array<Eigen::MatrixXd, d> reference_gradients;
            std::array<Eigen::MatrixXd, d> gradients_above;
            std::array<Eigen::MatrixXd, d> gradients_below;
            std::array<Eigen::MatrixXd, num_pairs> hessians;

            for (const auto& element : basis.grid()) {
                INFO("element " << element.index());
                const std::array<int, d>& first = element.first_active();
                const Eigen::MatrixXd points = points_on(element);

                // Each method fills the lower orders as the others do
                basis.hess_on_element(first, points, values, gradients,
                                      hessians);
                basis.grad_on_element(first, points, reference,
                                      reference_gradients);

                REQUIRE(values.isApprox(reference, 1e-14));

                for (std::size_t direction = 0; direction < d; ++direction)
                    REQUIRE(gradients[direction].isApprox(
                        reference_gradients[direction], 1e-14));

                basis.eval_on_element(first, points, reference);

                REQUIRE(values.isApprox(reference, 1e-14));

                for (std::size_t direction = 0; direction < d; ++direction) {
                    INFO("direction " << direction);
                    Eigen::MatrixXd shifted = points;

                    shifted.col(direction).array() += step;
                    basis.grad_on_element(first, shifted, above,
                                          gradients_above);
                    shifted.col(direction).array() -= 2. * step;
                    basis.grad_on_element(first, shifted, below,
                                          gradients_below);

                    REQUIRE(matches_difference(gradients[direction], above,
                                               below, step));

                    // Differentiate the gradient across the other direction
                    // of every pair holding this one
                    for (std::size_t index = 0; index < num_pairs; ++index) {
                        const auto [first_direction, second_direction] =
                            pairs[index];

                        if (first_direction != direction &&
                            second_direction != direction)
                            continue;

                        INFO("pair " << first_direction << ", "
                                     << second_direction);
                        const std::size_t other =
                            first_direction == direction ? second_direction
                                                         : first_direction;

                        REQUIRE(matches_difference(hessians[index],
                                                   gradients_above[other],
                                                   gradients_below[other],
                                                   step));
                    }
                }
            }
        }, entry);
    }
}

TEST_CASE("NURBS basis requires one positive weight per function",
          "[tensor_nurbs]")
{
    using Bivariate = TensorNURBS<double, 2>;

    const TensorBSpline<double, 2> bspline({quadratic(), cubic()});
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(bspline.num_functions());

    REQUIRE_THROWS_AS(Bivariate(bspline, weights.head(weights.size() - 1)),
                      std::invalid_argument);

    weights(4) = 0.;

    REQUIRE_THROWS_AS(Bivariate(bspline, weights), std::invalid_argument);
}
