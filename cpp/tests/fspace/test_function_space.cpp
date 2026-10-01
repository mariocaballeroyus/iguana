/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;
using iguana::CellType;
using iguana::Embedding;
using iguana::KnotVector;
using iguana::TensorBSpline;
using iguana::TensorDomain;

using Space = iguana::FunctionSpace<TensorBSpline<double, 2>>;

/// @brief Basis over uneven knot spans, with three elements along the first
///        direction and two along the second
TensorBSpline<double, 2> uneven()
{
    return TensorBSpline<double, 2>(TensorDomain<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., 1., 2., 6., 6., 6.}),
         KnotVector<double>(1, {0., 0., 1., 4., 4.})}));
}

/// @brief Inside cells, except those of the last column along the first
///        direction, which take the given type
Embedding<double, 2> with_last_column(CellType type)
{
    return Embedding<double, 2>({CellType::inside, CellType::inside, type,
                                 CellType::inside, CellType::inside, type});
}

/// @brief Integral of a univariate B-spline, (t_{i+p+1} - t_i) / (p + 1)
double integral(const KnotVector<double>& knot_vector, int function)
{
    const std::vector<double>& t = knot_vector.values();
    const int p = knot_vector.degree();

    return (t[function + p + 1] - t[function]) / (p + 1);
}

/// @brief Integral of the function of each degree of freedom over the
///        inside cells, assembled through the map of the space
Eigen::VectorXd integrals(const Space& space,
                          const Embedding<double, 2>& embedding)
{
    const TensorBSpline<double, 2>& basis = space.basis();

    // Two points per direction integrate the quadratic functions exactly
    iguana::DomainQuadrature<double, 2> quadrature;
    quadrature.fill(basis.domain(), embedding, CellType::inside,
                    iguana::GaussLegendre<double, 2>(2));

    Eigen::VectorXd result =
        Eigen::VectorXd::Zero(space.dof_map().num_dofs());
    Eigen::MatrixXd values;
    int held = 0;

    for (const auto& element : basis.domain()) {
        if (embedding.cell_type(element.index()) != CellType::inside)
            continue;

        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;
        ++held;

        basis.eval_on_element(element.first_active(),
                              quadrature.points().middleRows(first, count),
                              values);

        result(space.dof_map().dofs_on_element(element.index())) +=
            values * quadrature.weights().segment(first, count);
    }

    return result;
}

} // namespace

TEST_CASE("Integrating through the degrees of freedom gives the integral "
          "of every function", "[fspace]")
{
    const Embedding<double, 2> embedding = with_last_column(CellType::inside);
    const Space space(uneven(), embedding);
    const TensorBSpline<double, 2>& basis = space.basis();

    const Eigen::VectorXd computed = integrals(space, embedding);
    const int count = basis.axis(0).num_functions();

    REQUIRE(computed.size() == basis.num_functions());

    // With every cell inside, each function keeps its own index
    for (int function = 0; function < basis.num_functions(); ++function) {
        const double exact =
            integral(basis.axis(0).knots(), function % count)
            * integral(basis.axis(1).knots(), function / count);

        REQUIRE_THAT(computed(function), WithinRel(exact, 1e-12));
    }
}

TEST_CASE("Functions supported on outside cells alone get no degree of "
          "freedom", "[fspace]")
{
    const Embedding<double, 2> embedding = with_last_column(CellType::outside);
    const Space space(uneven(), embedding);
    const iguana::DofMap& dof_map = space.dof_map();

    // The last function along the first direction lives on the last column
    // alone, so each of its three products along the second is dropped
    REQUIRE(dof_map.num_dofs() == space.basis().num_functions() - 3);

    for (int element = 0; element < dof_map.num_elements(); ++element) {
        if (embedding.cell_type(element) == CellType::outside)
            REQUIRE(dof_map.dofs_on_element(element).empty());
    }

    // Every degree of freedom is non-zero inside, and the functions sum to
    // one over the inside cells, the parameters [0, 2] x [0, 4]
    const Eigen::VectorXd computed = integrals(space, embedding);

    REQUIRE((computed.array() > 0.).all());
    REQUIRE_THAT(computed.sum(), WithinRel(8., 1e-12));
}

TEST_CASE("A space needs one cell type per element", "[fspace]")
{
    REQUIRE_THROWS_AS(Space(uneven(), Embedding<double, 2>({CellType::inside})),
                      std::invalid_argument);
}
