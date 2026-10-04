/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::KnotVector;

std::array<double, 5> points_on(const KnotVector<double>& knots,
                                int element)
{
    const double start = knots.element_start(element);
    const double end = knots.element_end(element);

    return {start,
            .75 * start + .25 * end,
            .50 * start + .50 * end,
            .25 * start + .75 * end,
            end};
}

/// @brief Values of every function of a basis at points of one element,
///        with size (num_functions, num_points)
Eigen::MatrixXd all_values(const BSpline<double>& basis, int element,
                           std::span<const double> points)
{
    Eigen::MatrixXd values;
    basis.eval_on_element(basis.first_active(element), points, values);

    Eigen::MatrixXd result =
        Eigen::MatrixXd::Zero(basis.num_functions(), values.cols());
    result.middleRows(basis.first_active(element), basis.num_active()) =
        values;

    return result;
}

/// @brief Bernstein polynomials of a degree at a point of [0, 1]
Eigen::VectorXd bernstein(int degree, double t)
{
    Eigen::VectorXd result(degree + 1);
    double binomial = 1.;

    for (int index = 0; index <= degree; ++index) {
        result(index) = binomial * std::pow(t, index)
                        * std::pow(1. - t, degree - index);
        binomial = binomial * (degree - index) / (index + 1);
    }

    return result;
}

/// @brief Checks that the refinement matrix writes every coarse B-spline
///        in the fine ones, at points of every fine element
void check_refinement(const KnotVector<double>& coarse,
                      const KnotVector<double>& fine)
{
    const Eigen::MatrixXd refinement = iguana::refinement_matrix(coarse, fine);
    const BSpline<double> coarse_basis(coarse);
    const BSpline<double> fine_basis(fine);

    for (int element = 0; element < fine.num_elements(); ++element) {
        const std::array points = points_on(fine, element);

        // The coarse element holding the fine one
        int parent = 0;

        while (coarse.element_end(parent) < fine.element_end(element))
            ++parent;

        const Eigen::MatrixXd residual =
            refinement.transpose() * all_values(fine_basis, element, points)
            - all_values(coarse_basis, parent, points);

        REQUIRE_THAT(residual.cwiseAbs().maxCoeff(), WithinAbs(0., 1e-14));
    }
}

} // namespace

TEST_CASE("A knot vector rejects invalid definitions", "[knot_vector]")
{
    REQUIRE_THROWS_AS(KnotVector<double>(-1, {0., 0., 1., 1.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(KnotVector<double>(1, {0., 2., 1., 3.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(KnotVector<double>(2, {0., 0., 0., 1., 1.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(KnotVector<double>(1, {0., 0., 0., 0.}),
                      std::invalid_argument);
}

TEST_CASE("The elements are the non-empty knot spans of the parametric domain",
          "[knot_vector]")
{
    // Unclamped, with the parametric domain [1, 2] and an empty span at the
    // repeated knot 1.5, whose last repeat starts the second element
    const KnotVector<double> knots(2, {0., .5, 1., 1.5, 1.5, 2., 3., 3.5});

    REQUIRE(knots.domain_start() == 1.);
    REQUIRE(knots.domain_end() == 2.);
    REQUIRE(knots.num_elements() == 2);
    REQUIRE(knots.element_start(0) == 1.);
    REQUIRE(knots.element_end(0) == 1.5);
    REQUIRE(knots.element_start(1) == 1.5);
    REQUIRE(knots.element_end(1) == 2.);
    REQUIRE(knots.element_span(1) == 4);
}

TEST_CASE("Refinement writes the coarse functions in the fine ones",
          "[knot_vector]")
{
    // Degrees zero to three, clamped and unclamped, one with a repeated
    // interior knot
    const std::vector<KnotVector<double>> vectors{
        KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 4., 4., 4.}),
        KnotVector<double>(2, {0., 0., 0., 1., 1., 1.}),
        KnotVector<double>(3, {0., 0., 0., 0., 1., 2., 3., 3., 3., 3.}),
        KnotVector<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
        KnotVector<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5}),
        KnotVector<double>(0, {0., 1., 2., 3.})
    };

    for (const KnotVector<double>& coarse : vectors) {
        // Dyadic refinement, halving every element
        std::vector<double> midpoints;

        for (int element = 0; element < coarse.num_elements(); ++element)
            midpoints.push_back(
                (coarse.element_start(element) + coarse.element_end(element))
                / 2.);

        const KnotVector<double> fine = iguana::insert_knots(coarse, midpoints);

        REQUIRE(fine.num_elements() == 2 * coarse.num_elements());
        check_refinement(coarse, fine);
    }

    // Knots off the midpoints, in no order, one repeating a knot of the
    // vector and one inserted twice
    const KnotVector<double>& coarse = vectors.front();

    check_refinement(coarse, iguana::insert_knots(coarse, {2., .3, 3.7, 3.7}));
}

TEST_CASE("Refinement rejects knot vectors that do not nest",
          "[knot_vector]")
{
    const KnotVector<double> coarse(2, {0., 0., 0., 1., 2., 2., 2.});

    // A knot inserted at an end of the parametric domain
    REQUIRE_THROWS_AS(iguana::insert_knots(coarse, {2.}),
                      std::invalid_argument);

    // A fine knot vector of another degree, missing a knot of the coarse
    // one, or adding a knot at an end of the parametric domain
    REQUIRE_THROWS_AS(
        iguana::refinement_matrix(coarse,
                                  KnotVector<double>(1, {0., 0., 1., 2., 2.})),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        iguana::refinement_matrix(
            coarse, KnotVector<double>(2, {0., 0., 0., 1.5, 2., 2., 2.})),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        iguana::refinement_matrix(
            coarse, KnotVector<double>(2, {0., 0., 0., 1., 2., 2., 2., 2.})),
        std::invalid_argument);
}

TEST_CASE("Extraction writes the functions of each element in its Bernstein "
          "polynomials",
          "[knot_vector]")
{
    // Degrees zero to three, with interior knots repeated below the
    // degree, up to it and past it
    const std::vector<KnotVector<double>> vectors{
        KnotVector<double>(0, {0., 1., 2., 3.}),
        KnotVector<double>(1, {0., 0., .3, 1., 1.}),
        KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 4., 4., 4.}),
        KnotVector<double>(2, {0., 0., 0., .5, .5, 1., 1., 1.}),
        KnotVector<double>(2, {0., 0., 0., .5, .5, .5, 1., 1., 1.}),
        KnotVector<double>(3, {0., 0., 0., 0., .2, .2, .7, 1., 1., 1., 1.})
    };

    for (const KnotVector<double>& knots : vectors) {
        const BSpline<double> basis(knots);
        const std::vector<Eigen::MatrixXd> operators =
            iguana::extraction_operators(knots);

        REQUIRE(static_cast<int>(operators.size()) == knots.num_elements());

        for (int element = 0; element < knots.num_elements(); ++element) {
            const std::array points = points_on(knots, element);
            const double start = knots.element_start(element);
            const double length = knots.element_end(element) - start;

            Eigen::MatrixXd values;
            basis.eval_on_element(basis.first_active(element), points,
                                  values);

            for (std::size_t point = 0; point < points.size(); ++point) {
                const Eigen::VectorXd residual =
                    values.col(point)
                    - operators[element]
                          * bernstein(knots.degree(),
                                      (points[point] - start) / length);

                REQUIRE_THAT(residual.cwiseAbs().maxCoeff(),
                             WithinAbs(0., 1e-14));
            }
        }
    }

    // Unclamped, its end elements hold no Bernstein polynomials of a refined
    // basis
    REQUIRE_THROWS_AS(
        iguana::extraction_operators(KnotVector<double>(
            2, {0., .5, 1., 1.5, 1.5, 2., 3., 3.5})),
        std::invalid_argument);
}
