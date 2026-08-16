/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "iguana/basis/tensor_product.hpp"

#include<array>
#include<cmath>
#include<set>
#include<vector>

#include<catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;

template<int d>
using Tensor = iguana::TensorProductBSpline<d, double>;

/// @brief Univariate bases covering the knot vectors to handle.
BSpline<double> quadratic()
{
    return BSpline<double>(2, {0., 0., 0., 1., 2., 3., 3., 3.});
}

BSpline<double> cubic()
{
    return BSpline<double>(3, {0., 0., 0., 0., 1., 2., 2., 2., 2.});
}

BSpline<double> linear()
{
    return BSpline<double>(1, {0., 0., 1., 2., 2.});
}

BSpline<double> repeated()
{
    return BSpline<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.});
}

BSpline<double> uniform()
{
    return BSpline<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5});
}

/// @brief Points spread inside an element, its boundary left out so that
///        the differences of the derivative tests stay inside it.
template<int d>
Eigen::MatrixXd points_on(const Tensor<d>& basis, int element)
{
    const std::array<double, d> start = basis.element_start(element);
    const std::array<double, d> end = basis.element_end(element);

    Eigen::MatrixXd points(3, d);

    for (int k = 0; k < d; ++k)
        for (int q = 0; q < 3; ++q)
            points(q, k) = start[k]
                           + (end[k] - start[k]) * (.25 + .25 * q);

    return points;
}

/// @brief The directions differentiated by a slot of an order.
template<int d>
std::array<int, d> orders_of(int order, int slot)
{
    std::array<int, d> orders{};

    if (order == 0)
        return orders;

    if (order == 1) {
        orders[slot] = 1;
        return orders;
    }

    if (slot < d) {
        orders[slot] = 2;
        return orders;
    }

    // The mixed slots follow the lexicographic order of the pairs
    int next = d;

    for (int k = 0; k < d; ++k)
        for (int l = k + 1; l < d; ++l)
            if (next++ == slot) {
                orders[k] = 1;
                orders[l] = 1;
                return orders;
            }

    return orders;
}

/// @brief The multi-index of an active function of a tensor basis.
template<int d>
std::array<int, d> active_of(const Tensor<d>& basis, int row)
{
    std::array<int, d> each{};
    int remaining = row;

    for (int k = 0; k < d; ++k) {
        const int count = basis.axis(k).num_active();
        each[k] = remaining % count;
        remaining /= count;
    }

    return each;
}

} // namespace

TEST_CASE("The actives of an element are distinct and cover the basis",
          "[tensor_product]")
{
    const Tensor<2> basis({quadratic(), repeated()});

    Eigen::VectorXi actives;
    std::set<int> covered;

    for (int e = 0; e < basis.num_elements(); ++e) {
        basis.active_on_element(e, actives);

        REQUIRE(actives.size() == basis.num_active());

        std::set<int> seen;

        for (int r = 0; r < actives.size(); ++r) {
            REQUIRE(actives[r] >= 0);
            REQUIRE(actives[r] < basis.num_functions());

            seen.insert(actives[r]);
            covered.insert(actives[r]);
        }

        // No function is active twice on the same element
        REQUIRE(static_cast<int>(seen.size()) == basis.num_active());
    }

    // Every function of the basis is active somewhere
    REQUIRE(static_cast<int>(covered.size()) == basis.num_functions());
}

TEST_CASE("The element boxes fill the whole domain",
          "[tensor_product]")
{
    const Tensor<3> basis({quadratic(), repeated(), uniform()});

    double volume = 0.;

    for (int e = 0; e < basis.num_elements(); ++e) {
        const std::array<double, 3> start = basis.element_start(e);
        const std::array<double, 3> end = basis.element_end(e);

        double cell = 1.;

        for (int k = 0; k < 3; ++k) {
            REQUIRE(start[k] < end[k]);

            cell *= end[k] - start[k];
        }

        volume += cell;
    }

    // As many boxes as elements, none empty, together the whole domain
    double whole = 1.;

    for (int k = 0; k < 3; ++k) {
        const BSpline<double>& axis = basis.axis(k);

        whole *= axis.element_end(axis.num_elements() - 1)
                 - axis.element_start(0);
    }

    REQUIRE(std::abs(volume - whole) < 1e-12);
}

TEST_CASE("The basis is a partition of unity on every element",
          "[tensor_product]")
{
    const Tensor<3> basis({quadratic(), cubic(), linear()});

    Eigen::MatrixXd values;

    for (int e = 0; e < basis.num_elements(); ++e) {
        const Eigen::MatrixXd points = points_on(basis, e);

        basis.eval_on_element(basis.first_active(e), points, values);

        REQUIRE(values.rows() == basis.num_active());
        REQUIRE(values.cols() == points.rows());
        REQUIRE(values.minCoeff() >= 0.);

        for (Eigen::Index q = 0; q < values.cols(); ++q)
            REQUIRE(std::abs(values.col(q).sum() - 1.) < 1e-13);
    }
}

TEST_CASE("The values are the products over the directions",
          "[tensor_product]")
{
    const Tensor<2> basis({uniform(), cubic()});

    const Eigen::MatrixXd points = points_on(basis, 1);
    const std::array<int, 2> first = basis.first_active(1);

    Eigen::MatrixXd values;
    basis.eval_on_element(first, points, values);

    std::array<Eigen::MatrixXd, 2> axis_values;

    for (int k = 0; k < 2; ++k) {
        const std::vector<double> coords{points(0, k), points(1, k),
                                         points(2, k)};

        basis.axis(k).eval_on_element(first[k], coords, axis_values[k]);
    }

    // The row order of the evaluation is that of the actives, so the
    // first direction has to run fastest
    for (int r = 0; r < basis.num_active(); ++r) {
        const std::array<int, 2> each = active_of(basis, r);

        for (Eigen::Index q = 0; q < values.cols(); ++q)
            REQUIRE(std::abs(values(r, q) - axis_values[0](each[0], q)
                             * axis_values[1](each[1], q)) < 1e-15);
    }
}

TEST_CASE("The derivatives factorise, in the slot order of an order",
          "[tensor_product]")
{
    const Tensor<3> basis({quadratic(), cubic(), linear()});

    const Eigen::MatrixXd points = points_on(basis, 2);
    const std::array<int, 3> first = basis.first_active(2);

    std::vector<Eigen::MatrixXd> values;
    basis.eval_ders_on_element(first, points, 2, values);

    REQUIRE(static_cast<int>(values.size()) == 3);

    std::array<std::vector<Eigen::MatrixXd>, 3> axis_ders;

    for (int k = 0; k < 3; ++k) {
        const std::vector<double> coords{points(0, k), points(1, k),
                                         points(2, k)};

        basis.axis(k).eval_ders_on_element(first[k], coords, 2,
                                           axis_ders[k]);
    }

    for (int order = 0; order <= 2; ++order) {
        const int slots = basis.num_slots(order);
        const std::size_t entry = static_cast<std::size_t>(order);

        REQUIRE(values[entry].rows() == basis.num_active() * slots);

        for (int r = 0; r < basis.num_active(); ++r) {
            const std::array<int, 3> each = active_of(basis, r);

            for (int slot = 0; slot < slots; ++slot) {
                const std::array<int, 3> orders
                    = orders_of<3>(order, slot);

                for (Eigen::Index q = 0; q < points.rows(); ++q) {
                    double expected = 1.;

                    for (int k = 0; k < 3; ++k) {
                        const std::size_t of
                            = static_cast<std::size_t>(orders[k]);

                        expected *= axis_ders[k][of](each[k], q);
                    }

                    REQUIRE(std::abs(values[entry](r * slots + slot, q)
                                     - expected) < 1e-15);
                }
            }
        }
    }
}

TEST_CASE("The derivatives agree with central differences",
          "[tensor_product]")
{
    constexpr double step = 1e-6;

    const Tensor<2> basis({quadratic(), cubic()});

    std::vector<Eigen::MatrixXd> values;
    Eigen::MatrixXd before;
    Eigen::MatrixXd after;

    for (int e = 0; e < basis.num_elements(); ++e) {
        const Eigen::MatrixXd points = points_on(basis, e);
        const std::array<int, 2> first = basis.first_active(e);

        basis.eval_ders_on_element(first, points, 2, values);

        // The derivatives of a partition of unity vanish
        for (int order = 1; order <= 2; ++order) {
            const int slots = basis.num_slots(order);
            const std::size_t entry = static_cast<std::size_t>(order);

            for (int slot = 0; slot < slots; ++slot)
                for (Eigen::Index q = 0; q < points.rows(); ++q) {
                    double sum = 0.;

                    for (int r = 0; r < basis.num_active(); ++r)
                        sum += values[entry](r * slots + slot, q);

                    REQUIRE(std::abs(sum) < 1e-10);
                }
        }

        // The gradient is the difference quotient of the values
        for (int k = 0; k < 2; ++k) {
            Eigen::MatrixXd lower = points;
            Eigen::MatrixXd upper = points;

            lower.col(k).array() -= step;
            upper.col(k).array() += step;

            basis.eval_on_element(first, lower, before);
            basis.eval_on_element(first, upper, after);

            for (int r = 0; r < basis.num_active(); ++r)
                for (Eigen::Index q = 0; q < points.rows(); ++q) {
                    const double difference
                        = (after(r, q) - before(r, q)) / (2. * step);

                    REQUIRE(std::abs(difference - values[1](r * 2 + k, q))
                            < 1e-6);
                }
        }
    }
}

TEST_CASE("A single direction reproduces the univariate basis",
          "[tensor_product]")
{
    const BSpline<double> axis = cubic();
    const Tensor<1> basis({axis});

    Eigen::MatrixXd points(3, 1);
    points << .2, .7, 1.4;

    const std::vector<double> coords{.2, .7, 1.4};

    std::vector<Eigen::MatrixXd> values;
    std::vector<Eigen::MatrixXd> expected;

    basis.eval_ders_on_element({0}, points, 2, values);
    axis.eval_ders_on_element(0, coords, 2, expected);

    REQUIRE(values.size() == expected.size());

    for (std::size_t k = 0; k < values.size(); ++k)
        REQUIRE((values[k] - expected[k]).cwiseAbs().maxCoeff() < 1e-15);
}
