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

/// @brief Triangles of a convex polygon, given counterclockwise, extruded
///        along w over [bottom, top], counterclockwise seen from outside
std::vector<Eigen::Matrix3d> prism(const std::vector<Eigen::Vector2d>& polygon,
                                   double bottom, double top)
{
    std::vector<Eigen::Matrix3d> result;

    const auto at = [](const Eigen::Vector2d& vertex, double w) {
        return Eigen::Vector3d(vertex(0), vertex(1), w);
    };
    const auto add = [&result](const Eigen::Vector3d& first,
                               const Eigen::Vector3d& second,
                               const Eigen::Vector3d& third) {
        Eigen::Matrix3d triangle;
        triangle << first, second, third;
        result.push_back(triangle);
    };

    for (std::size_t vertex = 0; vertex < polygon.size(); ++vertex) {
        const Eigen::Vector2d& first = polygon[vertex];
        const Eigen::Vector2d& second = polygon[(vertex + 1) % polygon.size()];

        // The side of each edge, facing away from the polygon
        add(at(first, bottom), at(second, bottom), at(second, top));
        add(at(first, bottom), at(second, top), at(first, top));

        // The caps, fanned from the first vertex, the lower one facing down
        if (vertex >= 1 && vertex + 1 < polygon.size()) {
            add(at(polygon[0], top), at(first, top), at(second, top));
            add(at(polygon[0], bottom), at(second, bottom),
                at(first, bottom));
        }
    }

    return result;
}

/// @brief Moments of a convex polygon extruded along w over [bottom, top],
///        those of the polygon times the integrals of P_k over the interval
Eigen::VectorXd prism_moments(const std::vector<Eigen::Vector2d>& polygon,
                              double bottom, double top, int order)
{
    const Eigen::VectorXd base = polygon_moments(polygon, order);

    Eigen::MatrixXd antiderivatives;
    iguana::legendre_antiderivatives<double>(
        order, std::vector<double>{bottom, top}, antiderivatives);

    // The degree in w runs slowest
    Eigen::VectorXd result(base.size() * (order + 1));

    for (int degree = 0; degree <= order; ++degree)
        result.segment(degree * base.size(), base.size()) =
            (antiderivatives(degree, 1) - antiderivatives(degree, 0)) * base;

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

TEST_CASE("The moments of a cut cube integrate over its part inside",
          "[quadrature]")
{
    const int order = 3;

    const Eigen::Vector2d a(-1., -1.);
    const Eigen::Vector2d b(1., -1.);
    const Eigen::Vector2d c(1., 1.);
    const Eigen::Vector2d d(-1., 1.);

    // A slanted cut, u + v <= 1, by a prism reaching beyond the cube
    const Eigen::VectorXd slanted = iguana::reference_moments<double, 3>(
        prism({{-1., -1.}, {2., -1.}, {-1., 2.}}, -2., 2.), order);

    REQUIRE(slanted.isApprox(
        prism_moments({a, b, {1., 0.}, {0., 1.}, d}, -1., 1., order), 1e-13));

    // The lower half in w, whose faces u = -1 and u = 1 lie on the cube's
    const Eigen::VectorXd half = iguana::reference_moments<double, 3>(
        prism({a, b, c, d}, -1., 0.), order);

    REQUIRE(half.isApprox(prism_moments({a, b, c, d}, -1., 0., order),
                          1e-13));
}

TEST_CASE("A cell inside or outside the domain has all or none of its "
          "moments", "[quadrature]")
{
    const int order = 3;

    // By orthogonality, only the product of P_0 integrates to nonzero, the
    // area or volume of the cell
    const Eigen::VectorXd square = iguana::reference_moments<double, 2>(
        boundary({{-2., -2.}, {3., -2.}, {3., 3.}, {-2., 3.}}), order);

    REQUIRE_THAT(square(0), WithinAbs(4., 1e-14));
    REQUIRE(square.tail(square.size() - 1).isZero(1e-14));

    const Eigen::VectorXd cube = iguana::reference_moments<double, 3>(
        prism({{-2., -2.}, {3., -2.}, {3., 3.}, {-2., 3.}}, -2., 3.), order);

    REQUIRE_THAT(cube(0), WithinAbs(8., 1e-13));
    REQUIRE(cube.tail(cube.size() - 1).isZero(1e-13));

    // Domains away from the cell leave it without moments
    REQUIRE(iguana::reference_moments<double, 2>(
                boundary({{2., 2.}, {3., 2.}, {3., 3.}, {2., 3.}}), order)
                .isZero(1e-14));
    REQUIRE(iguana::reference_moments<double, 3>(
                prism({{2., 2.}, {3., 2.}, {3., 3.}, {2., 3.}}, 2., 3.), order)
                .isZero(1e-14));
}
