/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>
#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::BoxRule;
using Basis = iguana::TensorBSpline<double, 2>;
using Values = iguana::ElementValues<Basis, 2>;
using Stokes = iguana::StokesElement<Basis, 2>;
using Element = iguana::NavierStokesElement<Basis, 2>;
using Stabilized = iguana::NavierStokesStabilizedElement<Basis, 2>;
using StabilizedStokes = iguana::StokesStabilizedElement<Basis, 2>;

/// @brief Sides of the rectangle, unequal so that physical and parametric
///        derivatives differ
constexpr double width = 2.;
constexpr double height = 3.;

/// @brief Dynamic viscosity and stabilization constant of the elements
constexpr double viscosity = 1.5;
constexpr double stabilization = 3.;

/// @brief Knot vector of one element of [0, 1], whose B-splines are the
///        Bernstein polynomials of a degree
iguana::KnotVector<double> bernstein(int degree)
{
    std::vector<double> knots(degree + 1, 0.);
    knots.resize(2 * (degree + 1), 1.);

    return {degree, knots};
}

/// @brief Basis of one element of the unit square, of the given degrees
///        along each direction
Basis bernstein(int along_x, int along_y)
{
    const iguana::TensorGrid<double, 2> grid(
        {bernstein(along_x), bernstein(along_y)});

    return Basis{grid};
}

/// @brief Rectangle [0, width] x [0, height] of degree 2 with one element,
///        whose control points are the scaled Greville points, so that the
///        map scales each axis by its side
iguana::Patch<Basis, 2> rectangle()
{
    // The first direction runs fastest
    iguana::PointMatrix<double, 2> points(9, 2);

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i)
            points.row(i + 3 * j) << width * i / 2., height * j / 2.;
    }

    return {bernstein(2, 2), points};
}

/**
 * @brief Runs a check on the values of the unknowns at Gauss points of the
 *        rectangle, first in one field of three components, then in a field
 *        each, the components of the velocity one degree higher along their
 *        own direction, as divergence-conforming spaces take them
 *
 * @param check Check of the values, the positions of the points and their
 *        physical weights
 */
template<typename Check>
void for_each_layout(const Check& check)
{
    const iguana::Patch<Basis, 2> patch = rectangle();
    const Basis along_x = bernstein(3, 2);
    const Basis along_y = bernstein(2, 3);
    const iguana::ValueFlags flags{.physical_hessians = true};

    Values one_field(patch, flags);
    Values fields(patch, {along_x, along_y, patch.basis()}, flags);

    // Five points per direction integrate the cubic products of the
    // convection of cubic functions closely
    const std::array<double, 2> start{0., 0.};
    const std::array<double, 2> end{1., 1.};
    const iguana::GaussLegendre<double, 2> rule(5);
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.fill_to_reference_space(start, end, points, weights);
    BoxRule<double, 2>::map_to_parameter_space(start, end, points, weights);
    one_field.reinit(0, points);
    fields.reinit(0, points);

    // Physical weights, as an assembler gives them, and positions
    weights = weights.cwiseProduct(one_field.measures());
    Eigen::MatrixXd positions = points;
    positions.col(0) *= width;
    positions.col(1) *= height;

    {
        INFO("one field");
        check(one_field, positions, weights);
    }

    {
        INFO("a field per unknown");
        check(fields, positions, weights);
    }
}

/// @brief Number of rows of the local matrices, the functions of every
///        unknown
Eigen::Index num_rows(const Values& values)
{
    Eigen::Index rows = 0;

    for (std::size_t unknown = 0; unknown < 3; ++unknown)
        rows += values.values(values.num_fields() == 3 ? unknown : 0).rows();

    return rows;
}

/// @brief Coefficients of a flow far from rest, none of them alike
Eigen::VectorXd far_from_rest(Eigen::Index rows)
{
    Eigen::VectorXd result(rows);

    for (Eigen::Index row = 0; row < rows; ++row)
        result(row) = .8 * std::sin(1.3 * static_cast<double>(row) + .4);

    return result;
}

/// @brief Body force that varies over the points
Eigen::MatrixXd force(const Eigen::MatrixXd& positions)
{
    Eigen::MatrixXd result(positions.rows(), 2);
    result.col(0) = 1. + positions.col(1).array();
    result.col(1) = positions.col(0).array().square() - 2.;

    return result;
}

/// @brief Checks the tangent of an element against central differences of
///        its residual
template<typename E>
void check_tangent(const E& element, const Values& values,
                   const Eigen::MatrixXd& positions,
                   const Eigen::VectorXd& weights)
{
    const Eigen::Index rows = num_rows(values);
    const Eigen::VectorXd coefficients = far_from_rest(rows);
    const Eigen::MatrixXd source = force(positions);
    const double step = 1e-6;

    Eigen::MatrixXd tangent;
    element.local_tangent(values, weights, source, coefficients, tangent);

    REQUIRE(tangent.rows() == rows);
    REQUIRE(tangent.cols() == rows);

    Eigen::VectorXd ahead;
    Eigen::VectorXd behind;

    for (Eigen::Index column = 0; column < rows; ++column) {
        Eigen::VectorXd moved = coefficients;
        moved(column) += step;
        element.local_residual(values, weights, source, moved, ahead);
        moved(column) -= 2. * step;
        element.local_residual(values, weights, source, moved, behind);

        const Eigen::VectorXd difference = (ahead - behind) / (2. * step);

        REQUIRE((difference - tangent.col(column)).cwiseAbs().maxCoeff()
                < 1e-7 * tangent.cwiseAbs().maxCoeff());
    }
}

} // namespace

TEST_CASE("The tangents of the Navier-Stokes elements are the derivatives "
          "of their residuals", "[navier_stokes_element]")
{
    for_each_layout([](const Values& values, const Eigen::MatrixXd& positions,
                       const Eigen::VectorXd& weights) {
        check_tangent(Element(viscosity, stabilization), values, positions,
                      weights);
        check_tangent(Stabilized(viscosity, stabilization), values,
                      positions, weights);
    });
}

TEST_CASE("The Navier-Stokes elements at rest without a force are the "
          "Stokes elements", "[navier_stokes_element]")
{
    for_each_layout([](const Values& values, const Eigen::MatrixXd&,
                       const Eigen::VectorXd& weights) {
        const Eigen::Index rows = num_rows(values);
        const Eigen::VectorXd rest = Eigen::VectorXd::Zero(rows);
        const Eigen::MatrixXd none = Eigen::MatrixXd::Zero(weights.size(), 2);
        Eigen::MatrixXd tangent;
        Eigen::MatrixXd expected;

        Element(viscosity, stabilization)
            .local_tangent(values, weights, none, rest, tangent);
        Stokes(viscosity).local_tangent(values, weights, none, rest, expected);

        REQUIRE((tangent - expected).cwiseAbs().maxCoeff() < 1e-12);

        Stabilized(viscosity, stabilization)
            .local_tangent(values, weights, none, rest, tangent);
        StabilizedStokes(viscosity, stabilization)
            .local_tangent(values, weights, none, rest, expected);

        REQUIRE((tangent - expected).cwiseAbs().maxCoeff() < 1e-12);
    });
}

TEST_CASE("The stabilization of the Navier-Stokes elements vanishes where "
          "the momentum and continuity equations hold",
          "[navier_stokes_element]")
{
    for_each_layout([](const Values& values, const Eigen::MatrixXd& positions,
                       const Eigen::VectorXd& weights) {
        // The flow u = (x², −2xy) with p = x + y, free of divergence, and the
        // force f = (u · ∇)u − μ Δu + ∇p = (2x³ − 2μ + 1, 2x²y + 1) that
        // makes it solve the momentum equation, viscous term included
        const Eigen::ArrayXd x = positions.col(0).array();
        const Eigen::ArrayXd y = positions.col(1).array();
        const std::array<Eigen::VectorXd, 3> fields_of_flow{
            (x * x).matrix(), (-2. * x * y).matrix(), (x + y).matrix()};
        Eigen::MatrixXd body_force(weights.size(), 2);
        body_force.col(0) = (2. * x.cube() - 2. * viscosity + 1.).matrix();
        body_force.col(1) = (2. * x * x * y + 1.).matrix();

        const std::size_t fields = values.num_fields();
        const Eigen::Index rows = num_rows(values);
        Eigen::VectorXd coefficients(rows);
        Eigen::Index row = 0;

        // The functions at the points fit each polynomial exactly
        for (std::size_t unknown = 0; unknown < 3; ++unknown) {
            const Eigen::MatrixXd& functions =
                values.values(fields == 3 ? unknown : 0);

            coefficients.segment(row, functions.rows()) =
                functions.transpose().colPivHouseholderQr().solve(
                    fields_of_flow[unknown]);
            row += functions.rows();
        }

        Eigen::VectorXd residual;
        Eigen::VectorXd galerkin;

        Element(viscosity, stabilization)
            .local_residual(values, weights, body_force, coefficients,
                            residual);
        Stokes(viscosity).local_residual(values, weights, body_force,
                                         coefficients, galerkin);

        // With r_M = 0 and div u = 0, what is left over the Galerkin forms of
        // Stokes is the convection alone, ( (u · ∇)u, N_i e_a ) with
        // (u · ∇)u = (2x³, 2x²y), whatever the stabilization
        const std::array<Eigen::VectorXd, 2> convected{
            (2. * x.cube()).matrix(), (2. * x * x * y).matrix()};
        Eigen::VectorXd convection = Eigen::VectorXd::Zero(rows);
        row = 0;

        for (std::size_t a = 0; a < 2; ++a) {
            const Eigen::MatrixXd& functions =
                values.values(fields == 3 ? a : 0);

            convection.segment(row, functions.rows()) =
                functions * weights.cwiseProduct(convected[a]);
            row += functions.rows();
        }

        REQUIRE((residual - galerkin - convection).cwiseAbs().maxCoeff()
                < 1e-11 * convection.cwiseAbs().maxCoeff());

        Stabilized(viscosity, stabilization)
            .local_residual(values, weights, body_force, coefficients,
                            residual);

        REQUIRE((residual - galerkin - convection).cwiseAbs().maxCoeff()
                < 1e-11 * convection.cwiseAbs().maxCoeff());
    });
}

TEST_CASE("The Navier-Stokes elements need a positive viscosity and "
          "stabilization", "[navier_stokes_element]")
{
    REQUIRE_THROWS_AS(Element(0., stabilization), std::invalid_argument);
    REQUIRE_THROWS_AS(Element(viscosity, 0.), std::invalid_argument);
    REQUIRE_THROWS_AS(Stabilized(0., stabilization), std::invalid_argument);
    REQUIRE_THROWS_AS(Stabilized(viscosity, 0.), std::invalid_argument);
}
