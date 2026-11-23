/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;
using iguana::CellType;
using iguana::CellClassification;
using iguana::KnotVector;
using iguana::TensorBSpline;
using iguana::TensorGrid;

using Space = iguana::FunctionSpace<TensorBSpline<double, 2>>;
using MixedSpace = iguana::MixedFunctionSpace<TensorBSpline<double, 2>>;

/// @brief Basis over uneven knot spans, with three elements along the first
///        direction and two along the second
TensorBSpline<double, 2> uneven()
{
    return TensorBSpline<double, 2>(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., 1., 2., 6., 6., 6.}),
         KnotVector<double>(1, {0., 0., 1., 4., 4.})}));
}

/// @brief Basis on the elements of uneven(), cubic along the first direction
///        and linear along the second
TensorBSpline<double, 2> raised()
{
    return TensorBSpline<double, 2>(TensorGrid<double, 2>(
        {KnotVector<double>(3, {0., 0., 0., 0., 1., 2., 6., 6., 6., 6.}),
         KnotVector<double>(1, {0., 0., 1., 4., 4.})}));
}

/// @brief Inside cells, except those of the last column along the first
///        direction, which take the given type
CellClassification<double, 2> with_last_column(CellType type)
{
    return CellClassification<double, 2>(
        {CellType::inside, CellType::inside, type,
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
                          const CellClassification<double, 2>& classification)
{
    const TensorBSpline<double, 2>& basis = space.basis();

    // Two points per direction integrate the quadratic functions exactly
    iguana::QuadratureDomain<double, 2> quadrature;
    quadrature.fill(basis.grid(), classification, CellType::inside,
                    iguana::GaussLegendre<double, 2>(2));

    Eigen::VectorXd result =
        Eigen::VectorXd::Zero(space.dof_map().num_dofs());
    Eigen::MatrixXd values;
    int held = 0;

    for (const auto& element : basis.grid()) {
        if (classification.cell_type(element.index()) != CellType::inside)
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
    const CellClassification<double, 2> classification =
        with_last_column(CellType::inside);
    const Space space(uneven(), classification);
    const TensorBSpline<double, 2>& basis = space.basis();

    const Eigen::VectorXd computed = integrals(space, classification);
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
    const CellClassification<double, 2> classification =
        with_last_column(CellType::outside);
    const Space space(uneven(), classification);
    const iguana::DofMap& dof_map = space.dof_map();

    // The last function along the first direction lives on the last column
    // alone, so each of its three products along the second is dropped
    REQUIRE(dof_map.num_dofs() == space.basis().num_functions() - 3);

    for (int element = 0; element < dof_map.num_elements(); ++element) {
        if (classification.cell_type(element) == CellType::outside)
            REQUIRE(dof_map.dofs_on_element(element).empty());
    }

    // Every degree of freedom is non-zero inside, and the functions sum to
    // one over the inside cells, the parameters [0, 2] x [0, 4]
    const Eigen::VectorXd computed = integrals(space, classification);

    REQUIRE((computed.array() > 0.).all());
    REQUIRE_THAT(computed.sum(), WithinRel(8., 1e-12));
}

TEST_CASE("A block size gives each function adjacent degrees of freedom, "
          "listed component after component on an element", "[fspace]")
{
    const CellClassification<double, 2> classification =
        with_last_column(CellType::outside);
    const Space scalar(uneven(), classification);
    const Space vector(uneven(), classification, 3);

    REQUIRE(vector.block_size() == 3);
    REQUIRE(vector.num_dofs() == 3 * scalar.num_dofs());

    std::vector<int> dofs;

    // Component c of the k-th degree of freedom of the map is 3 k + c, and
    // outside cells list none
    for (int element = 0; element < scalar.dof_map().num_elements();
         ++element) {
        const std::span<const int> scalar_dofs =
            scalar.dof_map().dofs_on_element(element);
        const std::size_t count = scalar_dofs.size();
        vector.dofs_on_element(element, dofs);

        REQUIRE(dofs.size() == 3 * count);

        for (std::size_t component = 0; component < 3; ++component) {
            for (std::size_t function = 0; function < count; ++function)
                REQUIRE(dofs[component * count + function]
                        == 3 * scalar_dofs[function]
                               + static_cast<int>(component));
        }
    }
}

TEST_CASE("A space needs one cell type per element and a positive block "
          "size", "[fspace]")
{
    REQUIRE_THROWS_AS(
        Space(uneven(), CellClassification<double, 2>({CellType::inside})),
        std::invalid_argument);
    REQUIRE_THROWS_AS(Space(uneven(), with_last_column(CellType::inside), 0),
                      std::invalid_argument);
}

TEST_CASE("A mixed space numbers its fields one after another", "[fspace]")
{
    // The outside column drops functions of the first two fields, and the
    // second has two components
    const Space first(uneven(), with_last_column(CellType::outside));
    const Space second(raised(), with_last_column(CellType::outside), 2);
    const Space third(uneven(), with_last_column(CellType::inside));
    const MixedSpace mixed({first, second, third});

    REQUIRE(mixed.num_fields() == 3);
    REQUIRE(mixed.num_elements() == 6);

    // 12, 2 x 15 and 15 degrees of freedom, each field in a contiguous
    // range that starts where the previous one ends
    REQUIRE(mixed.num_dofs() == 57);

    int end = 0;

    for (std::size_t field = 0; field < mixed.num_fields(); ++field) {
        REQUIRE(mixed.offset(field) == end);
        end += mixed.field(field).num_dofs();
    }

    REQUIRE(end == mixed.num_dofs());

    // On an element, the degrees of freedom of each field in turn, shifted
    // into its range
    std::vector<int> dofs;
    std::vector<int> field_dofs;

    for (int element = 0; element < mixed.num_elements(); ++element) {
        INFO("element " << element);
        mixed.dofs_on_element(element, dofs);
        std::size_t listed = 0;

        for (std::size_t field = 0; field < mixed.num_fields(); ++field) {
            mixed.field(field).dofs_on_element(element, field_dofs);

            for (const int dof : field_dofs) {
                REQUIRE(listed < dofs.size());
                REQUIRE(dofs[listed++] == mixed.offset(field) + dof);
            }
        }

        REQUIRE(listed == dofs.size());
    }
}

TEST_CASE("A mixed space needs fields on the elements of the first",
          "[fspace]")
{
    const Space space(uneven(), with_last_column(CellType::inside));
    const Space moved(
        TensorBSpline<double, 2>(TensorGrid<double, 2>(
            {KnotVector<double>(2, {0., 0., 0., 1., 3., 6., 6., 6.}),
             KnotVector<double>(1, {0., 0., 1., 4., 4.})})),
        with_last_column(CellType::inside));

    REQUIRE_THROWS_AS(MixedSpace({}), std::invalid_argument);
    REQUIRE_THROWS_AS(MixedSpace({space, moved}), std::invalid_argument);
}

TEST_CASE("The divergence of hierarchical velocities lies in the "
          "hierarchical pressures and spans them", "[fspace]")
{
    using iguana::ElementValues;
    using iguana::HierarchicalGrid;
    using iguana::THBSpline;
    using Basis = THBSpline<double, 2>;
    using Region = iguana::Patch<Basis, 2>;

    // Six by five quadratic elements, so that the boxes of four leave a
    // partial one, refined twice: the velocities take one degree more
    // along their own direction and the pressure the degrees of the patch
    const TensorBSpline<double, 2> coarse(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 4., 5., 6., 6., 6.}),
         KnotVector<double>(2, {0., 0., 0., 1., 2., 3., 4., 5., 5., 5.})}));
    const HierarchicalGrid<double, 2> once =
        refine_admissible(HierarchicalGrid<double, 2>(coarse.grid()),
                          std::vector<int>{7}, true);
    const HierarchicalGrid<double, 2> grid = refine_admissible(
        once, std::vector<int>{once.num_elements() - 1}, true);
    REQUIRE(grid.num_levels() == 3);

    // The identity map, whose control points are the Greville points
    iguana::PointMatrix<double, 2> points(coarse.num_functions(), 2);

    for (int function = 0; function < coarse.num_functions(); ++function) {
        int rest = function;

        for (std::size_t direction = 0; direction < 2; ++direction) {
            const KnotVector<double>& knots =
                coarse.axis(direction).knots();
            const int count = coarse.axis(direction).num_functions();
            points(function, static_cast<Eigen::Index>(direction)) =
                greville_abscissae(knots)(rest % count);
            rest /= count;
        }
    }

    const Region patch =
        refine(iguana::Patch<TensorBSpline<double, 2>, 2>(coarse, points),
               grid);
    const CellClassification<double, 2> inside(
        std::vector<CellType>(grid.num_elements(), CellType::inside));

    const iguana::FunctionSpace<Basis> u_x(Basis(grid, {3, 2}), inside);
    const iguana::FunctionSpace<Basis> u_y(Basis(grid, {2, 3}), inside);
    const iguana::FunctionSpace<Basis> pressure(Basis(grid, {2, 2}), inside);
    const iguana::MixedFunctionSpace<Basis> mixed({u_x, u_y, pressure});

    iguana::QuadratureDomain<double, 2> quadrature;
    quadrature.fill(grid, inside, CellType::inside,
                    iguana::GaussLegendre<double, 2>(4));
    ElementValues<Basis, 2> values(patch, mixed.bases(),
                                   {.physical_gradients = true});

    // B_kj = (q_k, div v_j) over both velocities, M_kl = (q_k, q_l) and
    // G_ij = (div v_i, div v_j), assembled dense
    const int num_velocities = u_x.num_dofs() + u_y.num_dofs();
    const int num_pressures = pressure.num_dofs();
    Eigen::MatrixXd divergence =
        Eigen::MatrixXd::Zero(num_pressures, num_velocities);
    Eigen::MatrixXd mass = Eigen::MatrixXd::Zero(num_pressures, num_pressures);
    Eigen::MatrixXd gram =
        Eigen::MatrixXd::Zero(num_velocities, num_velocities);
    std::vector<int> dofs;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int element = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        values.reinit(element, quadrature.points().middleRows(first, count));
        const Eigen::VectorXd weights =
            quadrature.weights().segment(first, count).cwiseProduct(
                values.measures());

        // The divergence of each velocity function, d/dx of the first
        // component's functions and d/dy of the second's
        Eigen::MatrixXd divergences(0, count);

        for (std::size_t field = 0; field < 2; ++field) {
            const Eigen::MatrixXd& slope =
                values.physical_gradients(field)[field];
            divergences.conservativeResize(divergences.rows() + slope.rows(),
                                           Eigen::NoChange);
            divergences.bottomRows(slope.rows()) = slope;
        }

        mixed.dofs_on_element(element, dofs);
        const Eigen::Index num_velocity_dofs =
            divergences.rows();
        const Eigen::MatrixXd& pressures = values.values(2);

        for (Eigen::Index k = 0; k < pressures.rows(); ++k) {
            const int row =
                dofs[static_cast<std::size_t>(num_velocity_dofs + k)]
                - mixed.offset(2);

            for (Eigen::Index j = 0; j < num_velocity_dofs; ++j)
                divergence(row, dofs[static_cast<std::size_t>(j)]) +=
                    pressures.row(k).cwiseProduct(divergences.row(j))
                        .dot(weights);

            for (Eigen::Index l = 0; l < pressures.rows(); ++l)
                mass(row, dofs[static_cast<std::size_t>(num_velocity_dofs + l)]
                              - mixed.offset(2)) +=
                    pressures.row(k).cwiseProduct(pressures.row(l))
                        .dot(weights);
        }

        for (Eigen::Index i = 0; i < num_velocity_dofs; ++i) {
            for (Eigen::Index j = 0; j < num_velocity_dofs; ++j)
                gram(dofs[static_cast<std::size_t>(i)],
                     dofs[static_cast<std::size_t>(j)]) +=
                    divergences.row(i).cwiseProduct(divergences.row(j))
                        .dot(weights);
        }
    }

    // The projection of each divergence onto the pressures leaves nothing:
    // |div v_j|^2 = B_j^T M^-1 B_j
    const Eigen::LLT<Eigen::MatrixXd> factored(mass);
    const Eigen::MatrixXd projected =
        divergence.transpose() * factored.solve(divergence);

    for (int j = 0; j < num_velocities; ++j) {
        INFO("velocity function " << j);
        REQUIRE(gram(j, j) > 0.);
        REQUIRE(std::abs(gram(j, j) - projected(j, j)) < 1e-12 * gram(j, j));
    }

    // And, by the refinement in boxes, the divergences span every pressure
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomposition(divergence);
    decomposition.setThreshold(1e-10);
    REQUIRE(decomposition.rank() == num_pressures);
}
