/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;

std::vector<BSpline<double>> test_bases()
{
    return {
        BSpline<double>(2, {0., 0., 0., 1., 1., 1.}),
        BSpline<double>(2, {0., 0., 0., 1., 2., 3., 4., 4., 4.}),
        BSpline<double>(3, {0., 0., 0., 0., 1., 2., 3., 3., 3., 3.}),
        BSpline<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
        BSpline<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5}),
        BSpline<double>(0, {0., 1., 2., 3.})
    };
}

std::array<double, 5> points_on(const BSpline<double>& basis, int element)
{
    const double start = basis.knots().element_start(element);
    const double end = basis.knots().element_end(element);

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

/// @brief Checks that the refinement matrix writes every coarse function
///        in the fine ones, at points of every fine element
void check_refinement(const BSpline<double>& coarse,
                      const BSpline<double>& fine)
{
    const Eigen::MatrixXd refinement =
        iguana::refinement_matrix(coarse.knots(), fine.knots());

    for (int element = 0; element < fine.knots().num_elements(); ++element) {
        const std::array points = points_on(fine, element);

        // The coarse element holding the fine one
        int parent = 0;

        while (coarse.knots().element_end(parent)
               < fine.knots().element_end(element))
            ++parent;

        const Eigen::MatrixXd residual =
            refinement.transpose() * all_values(fine, element, points)
            - all_values(coarse, parent, points);

        REQUIRE_THAT(residual.cwiseAbs().maxCoeff(), WithinAbs(0., 1e-14));
    }
}

} // namespace

TEST_CASE("A basis rejects invalid definitions", "[bspline]")
{
    const std::vector<double> knots{0., 0., 0., 1., 1., 1.};

    REQUIRE_THROWS_AS(BSpline<double>(-1, knots), std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(BSpline<double>::max_degree + 1, knots),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(1, {0., 2., 1., 3.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(2, {0., 0., 0., 1., 1.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(1, {0., 0., 0., 0.}),
                      std::invalid_argument);
}

TEST_CASE("The basis is a partition of unity", "[bspline]")
{
    Eigen::MatrixXd values;

    for (const BSpline<double>& basis : test_bases()) {
        const int num_elements = basis.knots().num_elements();

        for (int element = 0; element < num_elements; ++element) {
            const std::array points = points_on(basis, element);

            basis.eval_on_element(basis.first_active(element), points, values);

            REQUIRE(values.rows() == basis.num_active());
            REQUIRE(values.cols()
                    == static_cast<Eigen::Index>(points.size()));
            REQUIRE(values.minCoeff() >= 0.);

            for (Eigen::Index point = 0; point < values.cols(); ++point) {
                REQUIRE_THAT(values.col(point).sum(), WithinAbs(1., 1e-14));
            }
        }
    }
}

TEST_CASE("An open quadratic basis is the Bernstein basis", "[bspline]")
{
    const BSpline<double> basis(2, {0., 0., 0., 1., 1., 1.});
    const std::array points{0., .25, .5, .75, 1.};

    Eigen::MatrixXd values;
    basis.eval_on_element(0, points, values);

    for (Eigen::Index point = 0; point < values.cols(); ++point) {
        const double u = points[static_cast<std::size_t>(point)];

        REQUIRE_THAT(values(0, point),
                     WithinAbs((1. - u) * (1. - u), 1e-14));
        REQUIRE_THAT(values(1, point),
                     WithinAbs(2. * u * (1. - u), 1e-14));
        REQUIRE_THAT(values(2, point), WithinAbs(u * u, 1e-14));
    }
}

TEST_CASE("Refinement writes the coarse functions in the fine ones",
          "[bspline]")
{
    for (const BSpline<double>& basis : test_bases()) {
        const iguana::KnotVector<double>& knots = basis.knots();

        // Dyadic refinement, halving every element
        std::vector<double> midpoints;

        for (int element = 0; element < knots.num_elements(); ++element)
            midpoints.push_back(
                (knots.element_start(element) + knots.element_end(element))
                / 2.);

        const BSpline<double> fine(iguana::insert_knots(knots, midpoints));

        REQUIRE(fine.knots().num_elements() == 2 * knots.num_elements());
        check_refinement(basis, fine);
    }

    // Knots off the midpoints, in no order, one repeating a knot of the
    // basis and one inserted twice
    const BSpline<double> basis(2, {0., 0., 0., 1., 2., 3., 4., 4., 4.});
    const BSpline<double> fine(
        iguana::insert_knots(basis.knots(), {2., .3, 3.7, 3.7}));

    check_refinement(basis, fine);
}

TEST_CASE("Refinement rejects bases that do not nest", "[bspline]")
{
    const BSpline<double> basis(2, {0., 0., 0., 1., 2., 2., 2.});

    // A knot inserted at an end of the parametric domain
    REQUIRE_THROWS_AS(iguana::insert_knots(basis.knots(), {2.}),
                      std::invalid_argument);

    // A fine basis of another degree, missing a knot of the coarse one, or
    // adding a knot at an end of the parametric domain
    using iguana::KnotVector;

    REQUIRE_THROWS_AS(
        iguana::refinement_matrix(basis.knots(),
                                  KnotVector<double>(1, {0., 0., 1., 2., 2.})),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        iguana::refinement_matrix(
            basis.knots(),
            KnotVector<double>(2, {0., 0., 0., 1.5, 2., 2., 2.})),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        iguana::refinement_matrix(
            basis.knots(),
            KnotVector<double>(2, {0., 0., 0., 1., 2., 2., 2., 2.})),
        std::invalid_argument);
}
