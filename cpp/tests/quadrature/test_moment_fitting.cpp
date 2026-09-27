/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/legendre.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;

/// @brief Tetrahedron x, y, z >= 0, x + y + z <= 3, counterclockwise seen
///        from outside
struct Tetrahedron
{
    Eigen::MatrixXd vertices = (Eigen::MatrixXd(4, 3) << 0., 0., 0., 3., 0.,
                                0., 0., 3., 0., 0., 0., 3.)
                                   .finished();
    Eigen::MatrixXi facets =
        (Eigen::MatrixXi(4, 3) << 0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3)
            .finished();
};

/// @brief Box [lower, upper], counterclockwise seen from outside
struct Box
{
    Box(const Eigen::Vector3d& lower, const Eigen::Vector3d& upper)
        : vertices(8, 3),
          facets((Eigen::MatrixXi(12, 3) << 0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7,
                  6, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2, 1,
                  3, 7, 1, 7, 5)
                     .finished())
    {
        // Corner c takes the upper bound in the directions of the bits of c
        for (int corner = 0; corner < 8; ++corner)
            for (int axis = 0; axis < 3; ++axis)
                vertices(corner, axis) =
                    (corner >> axis) & 1 ? upper(axis) : lower(axis);
    }

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi facets;
};

/// @brief Triangle x, y >= 0, x + y <= 3, counterclockwise
struct Triangle
{
    Eigen::MatrixXd vertices =
        (Eigen::MatrixXd(3, 2) << 0., 0., 3., 0., 0., 3.).finished();
    Eigen::MatrixXi facets =
        (Eigen::MatrixXi(3, 2) << 0, 1, 1, 2, 2, 0).finished();
};

/**
 * @brief Checks the rule of the cell [0.5, 1.5]^d of a solid x_k >= 0,
 *        sum x_k <= 3: it reproduces the moments of the part inside, with
 *        at most (order + 1)^d points of positive weight inside it
 */
template<std::size_t d, typename Solid>
void check_rule(const Solid& solid, int order)
{
    const iguana::MomentFitting<double, d> rule(solid.vertices, solid.facets,
                                                order);

    std::array<double, d> start;
    std::array<double, d> end;
    start.fill(0.5);
    end.fill(1.5);

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    rule.reference_rule(start, end, points, weights);

    // The facets on the reference cell, xi = 2 x - 2 in every direction
    std::vector<Eigen::Matrix<double, d, d>> facets(solid.facets.rows());

    for (Eigen::Index facet = 0; facet < solid.facets.rows(); ++facet)
        for (std::size_t corner = 0; corner < d; ++corner)
            facets[facet].col(corner) =
                2 * solid.vertices.row(solid.facets(facet, corner))
                        .transpose()
                        .array()
                - 2;

    Eigen::MatrixXd products;
    iguana::tensor_legendre_polynomials<double, d>(order, points, products);

    const Eigen::VectorXd moments =
        iguana::reference_moments<double, d>(facets, order);

    REQUIRE((products * weights).isApprox(moments, 1e-9));
    REQUIRE(weights.size() <= std::pow(order + 1, d));
    REQUIRE(weights.minCoeff() > 0.);

    // Inside the cell and, back in parameters, inside the solid
    REQUIRE(points.cwiseAbs().maxCoeff() <= 1.);
    REQUIRE((points.array() + 2).rowwise().sum().maxCoeff() / 2 <= 3. + 1e-12);
}

} // namespace

TEST_CASE("A fitted rule reproduces the moments of the part inside",
          "[quadrature]")
{
    check_rule<2>(Triangle{}, 3);
    check_rule<3>(Tetrahedron{}, 2);
}

TEST_CASE("A cell outside or barely inside the domain gets no points",
          "[quadrature]")
{
    const Tetrahedron solid;
    const iguana::MomentFitting<double, 3> rule(solid.vertices, solid.facets,
                                                2);

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.reference_rule({5., 5., 5.}, {6., 6., 6.}, points, weights);
    REQUIRE(weights.size() == 0);

    // A slab of half a thousandth of the cell, on the plane x = 0.25 where
    // the first candidates lie, which they could fit
    const Box slab({0.24975, -1., -1.}, {0.25025, 2., 2.});
    const iguana::MomentFitting<double, 3> thin(slab.vertices, slab.facets,
                                                2);

    thin.reference_rule({0., 0., 0.}, {1., 1., 1.}, points, weights);
    REQUIRE(weights.size() == 0);
}

TEST_CASE("A thin part keeps the better of its fits", "[quadrature]")
{
    // A slab of three thousandths of the cell around x = 0.25, which the
    // first candidates meet and the finer ones all miss
    const Box slab({0.2485, -1., -1.}, {0.2515, 2., 2.});
    const iguana::MomentFitting<double, 3> rule(slab.vertices, slab.facets,
                                                2);

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    rule.reference_rule({0., 0., 0.}, {1., 1., 1.}, points, weights);

    // Its measure on the reference cell, three thousandths of 2^3
    REQUIRE(weights.size() > 0);
    REQUIRE_THAT(weights.sum(), WithinRel(8. * 0.003, 1e-4));
}

TEST_CASE("Moment fitting fills the cut cells of a domain", "[quadrature]")
{
    // Three unit cells along x, inside, cut and outside the box x <= 1.5
    iguana::TensorBSpline<double, 3> basis(
        {iguana::BSpline<double>(1, {0., 0., 1., 2., 3., 3.}),
         iguana::BSpline<double>(1, {0., 0., 1., 1.}),
         iguana::BSpline<double>(1, {0., 0., 1., 1.})});

    const int num_functions = basis.num_functions();

    const iguana::TensorDomain<double, 3> domain(
        iguana::Patch<double, 3>(
            std::move(basis),
            iguana::PointMatrix<double>::Zero(num_functions, 3)),
        {iguana::CellType::inside, iguana::CellType::cut,
         iguana::CellType::outside});

    const Box solid({-1., -1., -1.}, {1.5, 2., 2.});

    iguana::DomainQuadrature<double, 3> quadrature;
    quadrature.fill(domain, iguana::CellType::inside,
                    iguana::GaussLegendre<double, 3>(2));
    quadrature.fill(domain, iguana::CellType::cut,
                    iguana::MomentFitting<double, 3>(solid.vertices,
                                                     solid.facets, 2));

    // Both cells integrate the part of the patch inside the box
    REQUIRE(quadrature.num_elements() == 2);
    REQUIRE_THAT(quadrature.weights().sum(), WithinRel(1.5, 1e-9));

    // The points of the cut cell lie in its part inside, 1 <= x <= 1.5
    const int first = quadrature.offsets()(1);
    const Eigen::VectorXd cut =
        quadrature.points().col(0).tail(quadrature.num_points() - first);

    REQUIRE(cut.minCoeff() >= 1.);
    REQUIRE(cut.maxCoeff() <= 1.5);
}

TEST_CASE("Moment fitting rejects malformed facets and orders",
          "[quadrature]")
{
    using Rule = iguana::MomentFitting<double, 3>;
    const Tetrahedron solid;

    // The vertices or facets do not have three columns
    REQUIRE_THROWS_AS(Rule(solid.vertices.leftCols(2), solid.facets, 2),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Rule(solid.vertices, solid.facets.leftCols(2), 2),
                      std::invalid_argument);

    // A facet refers to a missing vertex
    Eigen::MatrixXi missing = solid.facets;
    missing(0, 0) = 4;

    REQUIRE_THROWS_AS(Rule(solid.vertices, missing, 2),
                      std::invalid_argument);

    // The order lies outside [0, max_order]
    REQUIRE_THROWS_AS(Rule(solid.vertices, solid.facets, -1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Rule(solid.vertices, solid.facets, Rule::max_order + 1),
                      std::invalid_argument);
}
