/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/legendre.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/LU>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;

/// @brief Segments of a closed polygon given counterclockwise, each with
///        the polygon on its left
std::vector<Eigen::Matrix2d>
boundary(const std::vector<Eigen::Vector2d>& vertices)
{
    std::vector<Eigen::Matrix2d> result;

    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
        Eigen::Matrix2d segment;
        segment << vertices[vertex],
            vertices[(vertex + 1) % vertices.size()];
        result.push_back(segment);
    }

    return result;
}

/// @brief Moments of a convex polygon, by a Xiao-Gimbutas rule exact for
///        the Legendre products on each triangle from its first vertex
Eigen::VectorXd polygon_moments(const std::vector<Eigen::Vector2d>& vertices,
                                int order)
{
    const iguana::XiaoGimbutas<double> rule(2 * order);
    Eigen::VectorXd result = Eigen::VectorXd::Zero((order + 1) * (order + 1));

    for (std::size_t vertex = 1; vertex + 1 < vertices.size(); ++vertex) {
        const Eigen::Vector2d first_edge = vertices[vertex] - vertices[0];
        const Eigen::Vector2d second_edge = vertices[vertex + 1] - vertices[0];

        Eigen::Matrix2d edges;
        edges << first_edge, second_edge;

        Eigen::MatrixXd points =
            (rule.points() * edges.transpose()).rowwise()
            + vertices[0].transpose();

        Eigen::MatrixXd products;
        iguana::tensor_legendre_polynomials<double, 2>(order, points,
                                                       products);

        result += std::abs(edges.determinant()) * products * rule.weights();
    }

    return result;
}

} // namespace

TEST_CASE("The moments of a cut square integrate over its part inside",
          "[quadrature]")
{
    const int order = 4;

    const Eigen::Vector2d a(-1., -1.);
    const Eigen::Vector2d b(1., -1.);
    const Eigen::Vector2d c(1., 1.);
    const Eigen::Vector2d d(-1., 1.);

    // A slanted cut, u + v <= 1, by a triangle reaching beyond the square
    const Eigen::VectorXd slanted = iguana::reference_moments<double, 2>(
        boundary({{-1., -1.}, {2., -1.}, {-1., 2.}}), order);

    REQUIRE(slanted.isApprox(
        polygon_moments({a, b, {1., 0.}, {0., 1.}, d}, order), 1e-14));

    // The lower half, whose right side lies on the upper edge u = 1
    const Eigen::VectorXd half = iguana::reference_moments<double, 2>(
        boundary({a, b, {1., 0.}, {-1., 0.}}), order);

    REQUIRE(half.isApprox(
        polygon_moments({a, b, {1., 0.}, {-1., 0.}}, order), 1e-14));
}

TEST_CASE("A square inside or outside the domain has all or none of its "
          "moments", "[quadrature]")
{
    const int order = 3;

    // By orthogonality, only P_0(u) P_0(v) integrates to nonzero, the area
    const Eigen::VectorXd inside = iguana::reference_moments<double, 2>(
        boundary({{-2., -2.}, {3., -2.}, {3., 3.}, {-2., 3.}}), order);

    REQUIRE_THAT(inside(0), WithinAbs(4., 1e-14));
    REQUIRE(inside.tail(inside.size() - 1).isZero(1e-14));

    const Eigen::VectorXd outside = iguana::reference_moments<double, 2>(
        boundary({{2., 2.}, {3., 2.}, {3., 3.}, {2., 3.}}), order);

    REQUIRE(outside.isZero(1e-14));
}
