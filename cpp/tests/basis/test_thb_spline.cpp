/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using Eigen::placeholders::all;
using iguana::HierarchicalGrid;
using iguana::KnotVector;
using iguana::TensorBSpline;
using iguana::TensorGrid;
using iguana::THBSpline;

/// @brief Quadratic grid of 7 x 6 elements on [0, 7] x [0, 6], whose boxes
///        of four elements leave a partial box along the first direction
HierarchicalGrid<double, 2> coarse()
{
    return HierarchicalGrid<double, 2>(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 7.,
                                7.}),
         KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 4., 5., 6., 6.,
                                6.})}));
}

/// @brief The grid refined twice around its last element, three levels
///        with a graded transition
HierarchicalGrid<double, 2> refined()
{
    const HierarchicalGrid<double, 2> once =
        refine_admissible(coarse(), std::vector<int>{41}, true);

    return refine_admissible(once, std::vector<int>{once.num_elements() - 1},
                             true);
}

/// @brief Points inside an active element, at fractions of its extent
Eigen::MatrixXd points_in(const HierarchicalGrid<double, 2>& grid,
                          int element)
{
    const std::array<double, 2> start = grid.element_start(element);
    const std::array<double, 2> end = grid.element_end(element);
    const std::array fractions{.15, .5, .85};
    Eigen::MatrixXd points(fractions.size(), 2);

    for (std::size_t direction = 0; direction < 2; ++direction) {
        for (std::size_t point = 0; point < fractions.size(); ++point)
            points(point, direction) =
                start[direction]
                + fractions[(point + direction) % 3]
                      * (end[direction] - start[direction]);
    }

    return points;
}

/// @brief Level of a function of a basis
int level_of(const THBSpline<double, 2>& basis, int function)
{
    int level = 0;

    while (function >= basis.offset(level + 1))
        ++level;

    return level;
}

} // namespace

TEST_CASE("An unrefined hierarchical basis is the tensor basis", "[thb_spline]")
{
    const THBSpline<double, 2> basis(coarse(), {3, 2});
    const TensorBSpline<double, 2> tensor = basis.level(0);

    REQUIRE(basis.num_levels() == 1);
    REQUIRE(basis.num_functions() == tensor.num_functions());
    REQUIRE(tensor.axis(0).degree() == 3);
    REQUIRE(tensor.axis(1).degree() == 2);

    Eigen::VectorXi actives;
    Eigen::VectorXi tensor_actives;
    Eigen::MatrixXd values;
    Eigen::MatrixXd tensor_values;
    std::array<Eigen::MatrixXd, 2> gradients;
    std::array<Eigen::MatrixXd, 2> tensor_gradients;

    for (int element = 0; element < basis.grid().num_elements(); ++element) {
        const Eigen::MatrixXd points = points_in(basis.grid(), element);
        basis.active_on_element(element, actives);
        tensor.active_on_element(element, tensor_actives);
        basis.grad_on_element(element, points, values, gradients);
        tensor.grad_on_element(element, points, tensor_values,
                               tensor_gradients);

        REQUIRE(actives == tensor_actives);
        REQUIRE(values == tensor_values);
        REQUIRE(gradients == tensor_gradients);
    }
}

TEST_CASE("Truncated hierarchical functions partition unity and come from "
          "two consecutive levels", "[thb_spline]")
{
    const HierarchicalGrid<double, 2> grid = refined();
    REQUIRE(grid.num_levels() == 3);

    for (const std::array<int, 2> degrees : {std::array{3, 2}, std::array{2, 3},
                                             std::array{2, 2}}) {
        INFO("degrees " << degrees[0] << ", " << degrees[1]);
        const THBSpline<double, 2> basis(grid, degrees);

        // Every function belongs to some element, and the functions of a
        // level are sorted
        std::set<int> seen;
        Eigen::VectorXi actives;
        Eigen::MatrixXd values;
        std::array<Eigen::MatrixXd, 2> gradients;

        for (int element = 0; element < grid.num_elements(); ++element) {
            INFO("element " << element);
            const Eigen::MatrixXd points = points_in(grid, element);
            basis.active_on_element(element, actives);
            basis.grad_on_element(element, points, values, gradients);

            REQUIRE(std::ranges::is_sorted(actives));
            REQUIRE(values.rows() == actives.size());
            REQUIRE(values.minCoeff() >= 0.);

            // The levels of the functions are the element's and the one
            // before at most, by the grading of the grid
            const int top = grid.level_of(element);

            for (const int function : actives) {
                seen.insert(function);
                const int level = level_of(basis, function);
                REQUIRE(level <= top);
                REQUIRE(level >= top - 1);
            }

            for (Eigen::Index point = 0; point < points.rows(); ++point) {
                REQUIRE_THAT(values.col(point).sum(), WithinAbs(1., 1e-13));

                for (std::size_t direction = 0; direction < 2; ++direction)
                    REQUIRE_THAT(gradients[direction].col(point).sum(),
                                 WithinAbs(0., 1e-12));
            }
        }

        REQUIRE(static_cast<int>(seen.size()) == basis.num_functions());
    }
}

TEST_CASE("Truncated hierarchical functions preserve the coefficients of "
          "coarse splines", "[thb_spline]")
{
    const HierarchicalGrid<double, 2> grid = refined();
    const THBSpline<double, 2> basis(grid, {3, 2});
    const TensorBSpline<double, 2>& tensor = basis.level(0);

    std::mt19937 generator(7);
    std::uniform_real_distribution<double> uniform(-1., 1.);
    Eigen::MatrixXd coarse(tensor.num_functions(), 2);

    for (Eigen::Index k = 0; k < coarse.size(); ++k)
        coarse(k) = uniform(generator);

    const Eigen::MatrixXd coefficients = basis.coefficients(coarse);
    REQUIRE(coefficients.rows() == basis.num_functions());

    // A function of the basis on level 0 keeps its coarse coefficient
    for (std::size_t k = 0; k < basis.active_functions(0).size(); ++k)
        REQUIRE(coefficients.row(static_cast<Eigen::Index>(k))
                == coarse.row(basis.active_functions(0)[k]));

    // The spline is the same on every element, with the same derivatives
    Eigen::VectorXi actives;
    Eigen::VectorXi tensor_actives;
    Eigen::MatrixXd values;
    Eigen::MatrixXd tensor_values;
    std::array<Eigen::MatrixXd, 2> gradients;
    std::array<Eigen::MatrixXd, 2> tensor_gradients;

    for (int element = 0; element < grid.num_elements(); ++element) {
        INFO("element " << element);
        const Eigen::MatrixXd points = points_in(grid, element);
        basis.active_on_element(element, actives);
        basis.grad_on_element(element, points, values, gradients);

        const int coarse_element =
            element_holding(tensor.grid(), {points(0, 0), points(0, 1)});
        tensor.active_on_element(coarse_element, tensor_actives);
        tensor.grad_on_element(coarse_element, points, tensor_values,
                               tensor_gradients);

        const Eigen::MatrixXd spline =
            values.transpose() * coefficients(actives, all);
        const Eigen::MatrixXd expected =
            tensor_values.transpose() * coarse(tensor_actives, all);
        REQUIRE((spline - expected).cwiseAbs().maxCoeff() < 1e-13);

        for (std::size_t direction = 0; direction < 2; ++direction) {
            const Eigen::MatrixXd slope = gradients[direction].transpose()
                                          * coefficients(actives, all);
            const Eigen::MatrixXd expected_slope =
                tensor_gradients[direction].transpose()
                * coarse(tensor_actives, all);
            REQUIRE((slope - expected_slope).cwiseAbs().maxCoeff() < 1e-12);
        }
    }

    REQUIRE_THROWS_AS(basis.coefficients(coarse.topRows(3)),
                      std::invalid_argument);
}

TEST_CASE("Hierarchical gradients and Hessians match finite differences",
          "[thb_spline]")
{
    const HierarchicalGrid<double, 2> grid = refined();
    const THBSpline<double, 2> basis(grid, {3, 3});
    const double step = 1e-5;

    Eigen::MatrixXd values;
    std::array<Eigen::MatrixXd, 2> gradients;
    std::array<Eigen::MatrixXd, 3> hessians;
    Eigen::MatrixXd forward;
    Eigen::MatrixXd backward;
    std::array<Eigen::MatrixXd, 2> forward_gradients;
    std::array<Eigen::MatrixXd, 2> backward_gradients;

    // A coarse element, a fine one and one between them
    for (const int element : {0, grid.offset(1) + 1, grid.num_elements() - 1}) {
        INFO("element " << element);
        const Eigen::MatrixXd points = points_in(grid, element).row(1);
        basis.hess_on_element(element, points, values, gradients, hessians);

        for (std::size_t direction = 0; direction < 2; ++direction) {
            Eigen::MatrixXd ahead = points;
            Eigen::MatrixXd behind = points;
            ahead(0, direction) += step;
            behind(0, direction) -= step;

            basis.grad_on_element(element, ahead, forward, forward_gradients);
            basis.grad_on_element(element, behind, backward,
                                  backward_gradients);

            const Eigen::MatrixXd difference =
                (forward - backward) / (2 * step);
            REQUIRE((difference - gradients[direction]).cwiseAbs().maxCoeff()
                    < 1e-8);

            // The row of the Hessian along the direction, pure and mixed
            for (std::size_t other = 0; other < 2; ++other) {
                const Eigen::MatrixXd second =
                    (forward_gradients[other] - backward_gradients[other])
                    / (2 * step);
                const std::size_t pair =
                    iguana::pair_index<2>(direction, other);
                REQUIRE((second - hessians[pair]).cwiseAbs().maxCoeff()
                        < 1e-7);
            }
        }
    }
}

TEST_CASE("The boundary functions are those non-zero on the sides",
          "[thb_spline]")
{
    const HierarchicalGrid<double, 2> grid = refined();
    const THBSpline<double, 2> basis(grid, {3, 2});
    Eigen::VectorXi actives;
    Eigen::MatrixXd values;

    for (std::size_t direction = 0; direction < 2; ++direction) {
        const Eigen::VectorXi boundary = basis.boundary_functions(direction);
        REQUIRE(std::ranges::is_sorted(boundary));

        // The functions non-zero at points on the two sides normal to the
        // direction, of the elements touching them
        const double side = direction == 0 ? 7. : 6.;
        std::set<int> non_zero;

        for (int element = 0; element < grid.num_elements(); ++element) {
            const std::array<double, 2> start = grid.element_start(element);
            const std::array<double, 2> end = grid.element_end(element);
            Eigen::MatrixXd points = points_in(grid, element);

            if (start[direction] == 0.)
                points.col(direction).setZero();
            else if (end[direction] == side)
                points.col(direction).setConstant(side);
            else
                continue;

            basis.active_on_element(element, actives);
            basis.eval_on_element(element, points, values);

            for (Eigen::Index k = 0; k < actives.size(); ++k) {
                if (values.row(k).cwiseAbs().maxCoeff() > 1e-13)
                    non_zero.insert(actives(k));
            }
        }

        REQUIRE(std::vector<int>(non_zero.begin(), non_zero.end())
                == std::vector<int>(boundary.begin(), boundary.end()));
    }
}

TEST_CASE("A hierarchical basis rejects what it cannot be built on",
          "[thb_spline]")
{
    REQUIRE_THROWS_AS((THBSpline<double, 2>(coarse(), {-1, 2})),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((THBSpline<double, 2>(coarse(), {2, 11})),
                      std::invalid_argument);

    // Unclamped coarse knots
    const HierarchicalGrid<double, 2> unclamped(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 1., 2., 3., 4., 5., 6., 7.}),
         KnotVector<double>(2, {0., 0., 0., 1., 2., 2., 2.})}));
    REQUIRE_THROWS_AS((THBSpline<double, 2>(unclamped, {2, 2})),
                      std::invalid_argument);
}
