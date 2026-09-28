/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;
using iguana::BSpline;
using iguana::CellType;
using iguana::Embedding;
using iguana::TensorBSpline;

using Quadrature = iguana::DomainQuadrature<double, 3>;
using Gauss = iguana::GaussLegendre<double, 3>;

/// @brief Basis over uneven knot spans, with six elements
TensorBSpline<double, 3> uneven()
{
    return TensorBSpline<double, 3>(
        {BSpline<double>(2, {0., 0., 0., 1., 3., 4., 4., 4.}),
         BSpline<double>(1, {0., 0., .5, 2., 2.}),
         BSpline<double>(1, {0., 0., 1., 1.})});
}

/// @brief Inside cells 0, 2 and 4, cut cells 1 and 5 and outside cell 3
Embedding<double, 3> mixed()
{
    return Embedding<double, 3>({CellType::inside, CellType::cut,
                                 CellType::inside, CellType::outside,
                                 CellType::inside, CellType::cut});
}

/// @brief Integer vector from its entries
Eigen::VectorXi integers(const std::vector<int>& entries)
{
    return Eigen::Map<const Eigen::VectorXi>(entries.data(), entries.size());
}

} // namespace

TEST_CASE("The flat arrays must describe one block of points per element",
          "[quadrature]")
{
    const Eigen::VectorXi elements = integers({3, 7});
    const Eigen::VectorXi offsets = integers({0, 5, 5});
    const Eigen::MatrixXd points = Eigen::MatrixXd::Zero(5, 3);
    const Eigen::VectorXd weights = Eigen::VectorXd::Ones(5);

    // An element may hold no points
    REQUIRE_NOTHROW(Quadrature(elements, offsets, points, weights));

    REQUIRE_THROWS_AS(Quadrature(elements, offsets,
                                 Eigen::MatrixXd::Zero(5, 2), weights),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Quadrature(elements, offsets, points,
                                 Eigen::VectorXd::Ones(4)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Quadrature(elements, integers({0, 5}), points,
                                 weights),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Quadrature(elements, integers({0, 6, 5}), points,
                                 weights),
                      std::invalid_argument);
}

TEST_CASE("The filled cells integrate the polynomials of the rule exactly",
          "[quadrature]")
{
    const TensorBSpline<double, 3> basis = uneven();
    const Embedding<double, 3> embedding = mixed();

    // Exact up to degree 2n - 1 in each direction
    const std::array<int, 3> counts{2, 3, 1};

    Quadrature quadrature;
    quadrature.fill(basis, embedding, CellType::inside, Gauss(counts));

    REQUIRE(quadrature.elements() == integers({0, 2, 4}));

    const Eigen::MatrixXd& points = quadrature.points();
    const Eigen::VectorXd& weights = quadrature.weights();
    int held = 0;

    for (iguana::TensorDomainIterator<double, 3> element(basis);
         element != std::default_sentinel; ++element) {
        if (embedding.cell_type(element.index()) != CellType::inside)
            continue;

        const int first = quadrature.offsets()(held);
        const int last = quadrature.offsets()(held + 1);
        ++held;

        for (int a = 0; a < 2 * counts[0]; ++a) {
            for (int b = 0; b < 2 * counts[1]; ++b) {
                for (int c = 0; c < 2 * counts[2]; ++c) {
                    const std::array<int, 3> degrees{a, b, c};
                    double sum = 0.;
                    double exact = 1.;

                    for (int point = first; point < last; ++point)
                        sum += weights(point)
                               * std::pow(points(point, 0), a)
                               * std::pow(points(point, 1), b)
                               * std::pow(points(point, 2), c);

                    for (std::size_t direction = 0; direction < 3;
                         ++direction) {
                        const int power = degrees[direction] + 1;

                        exact *= (std::pow(element.end()[direction], power)
                                  - std::pow(element.start()[direction],
                                             power))
                                 / power;
                    }

                    REQUIRE_THAT(sum, WithinRel(exact, 1e-12));
                }
            }
        }
    }
}

TEST_CASE("Cell types are filled one after another, once each",
          "[quadrature]")
{
    const TensorBSpline<double, 3> basis = uneven();
    const Embedding<double, 3> embedding = mixed();

    Quadrature quadrature;
    quadrature.fill(basis, embedding, CellType::inside, Gauss(2));
    quadrature.fill(basis, embedding, CellType::cut, Gauss(1));

    REQUIRE(quadrature.elements() == integers({0, 2, 4, 1, 5}));
    REQUIRE(quadrature.offsets() == integers({0, 8, 16, 24, 25, 26}));

    // A cell type already held, or an embedding without one cell type per
    // element, is rejected, and nothing changes
    const Quadrature before = quadrature;

    REQUIRE_THROWS_AS(
        quadrature.fill(basis, embedding, CellType::inside, Gauss(1)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        quadrature.fill(basis, Embedding<double, 3>({CellType::outside}),
                        CellType::outside, Gauss(1)),
        std::invalid_argument);
    REQUIRE(quadrature.offsets() == before.offsets());
    REQUIRE(quadrature.points() == before.points());
}
