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
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <unsupported/Eigen/KroneckerProduct>

namespace
{

using Catch::Matchers::WithinRel;
using Eigen::kroneckerProduct;
using iguana::BoxRule;
using Basis = iguana::TensorBSpline<double, 2>;
using Values = iguana::ElementValues<Basis, 2>;
using Element = iguana::StokesElement<Basis, 2>;
using Stabilized = iguana::StokesStabilizedElement<Basis, 2>;

/// @brief Sides of the rectangle, unequal so that physical and parametric
///        derivatives differ
constexpr double width = 2.;
constexpr double height = 3.;

/// @brief Dynamic viscosity and stabilization constant of the elements
constexpr double viscosity = 1.5;
constexpr double stabilization = 3.;

/// @brief Degrees along each direction of the basis of each unknown, the
///        components of the velocity followed by the pressure
using Degrees = std::array<std::array<int, 2>, 3>;

/// @brief Term c x^a y^b of a polynomial
struct Monomial
{
    double coefficient;
    int x;
    int y;
};

/// @brief Polynomial in physical space, a sum of monomials
using Polynomial = std::vector<Monomial>;

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

/// @brief Coefficients of t^k in the Bernstein polynomials of a degree p,
///        their blossoms C(i, k) / C(p, k)
Eigen::VectorXd power(int degree, int k)
{
    Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(degree + 1);

    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j < k; ++j)
            coefficients(i) *= static_cast<double>(i - j) / (degree - j);
    }

    return coefficients;
}

/// @brief Coefficients of a polynomial in the basis of the rectangle of the
///        given degrees, the first direction running fastest
Eigen::VectorXd coefficients(const std::array<int, 2>& degrees,
                             const Polynomial& polynomial)
{
    Eigen::VectorXd result =
        Eigen::VectorXd::Zero((degrees[0] + 1) * (degrees[1] + 1));

    for (const Monomial& term : polynomial)
        result += term.coefficient
                  * kroneckerProduct(
                        std::pow(height, term.y) * power(degrees[1], term.y),
                        std::pow(width, term.x) * power(degrees[0], term.x));

    return result;
}

/// @brief Local coefficients of the velocity and the pressure, one unknown
///        after another, each in the basis of its degrees
Eigen::VectorXd local(const Degrees& degrees, const Polynomial& u_x,
                      const Polynomial& u_y, const Polynomial& p)
{
    const Eigen::VectorXd x = coefficients(degrees[0], u_x);
    const Eigen::VectorXd y = coefficients(degrees[1], u_y);
    const Eigen::VectorXd pressure = coefficients(degrees[2], p);

    Eigen::VectorXd result(x.size() + y.size() + pressure.size());
    result << x, y, pressure;

    return result;
}

/// @brief Local coefficients of the linear velocity A x and no pressure
Eigen::VectorXd linear(const Degrees& degrees, const Eigen::Matrix2d& a)
{
    return local(degrees, {{a(0, 0), 1, 0}, {a(0, 1), 0, 1}},
                 {{a(1, 0), 1, 0}, {a(1, 1), 0, 1}}, {});
}

/// @brief Number of functions of an unknown
Eigen::Index num_functions(const Degrees& degrees, std::size_t unknown)
{
    return (degrees[unknown][0] + 1) * (degrees[unknown][1] + 1);
}

/// @brief Number of rows of the local matrices, the functions of every
///        unknown
Eigen::Index num_rows(const Degrees& degrees)
{
    return num_functions(degrees, 0) + num_functions(degrees, 1)
           + num_functions(degrees, 2);
}

/**
 * @brief Runs a check on the values of the unknowns at Gauss points of the
 *        rectangle, first in one field of three components, then in a field
 *        each, the components of the velocity one degree higher along their
 *        own direction, as divergence-conforming spaces take them
 *
 * @param check Check of the values, the points, the physical weights and
 *        the degrees of the basis of each unknown
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

    // Four points per direction integrate the products of cubic functions
    const std::array<double, 2> start{0., 0.};
    const std::array<double, 2> end{1., 1.};
    const iguana::GaussLegendre<double, 2> rule(4);
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.fill_to_reference_space(start, end, points, weights);
    BoxRule<double, 2>::map_to_parameter_space(start, end, points, weights);
    one_field.reinit(0, points);
    fields.reinit(0, points);

    // Physical weights, as an assembler gives them
    weights = weights.cwiseProduct(one_field.measures());

    {
        INFO("one field");
        check(one_field, points, weights, Degrees{{{2, 2}, {2, 2}, {2, 2}}});
    }

    {
        INFO("a field per unknown");
        check(fields, points, weights, Degrees{{{3, 2}, {2, 3}, {2, 2}}});
    }
}

/// @brief Velocity gradients of the linear velocities u = A x and v = B x
const Eigen::Matrix2d a{{1., -2.}, {3., .5}};
const Eigen::Matrix2d b{{-1., 2.}, {.5, 4.}};

/// @brief Checks the Galerkin forms against their exact integrals
void check_galerkin(const Values& values, const Eigen::MatrixXd&,
                    const Eigen::VectorXd& weights, const Degrees& degrees)
{
    const Element element(viscosity);
    const double area = width * height;
    const Eigen::Index rows = num_rows(degrees);
    const Eigen::Index pressures = num_functions(degrees, 2);

    // A constant body force, and the tangent at zero coefficients, the
    // stiffness
    Eigen::MatrixXd source(weights.size(), 2);
    source.col(0).setConstant(.7);
    source.col(1).setConstant(-1.2);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(rows);

    Eigen::MatrixXd stiffness;
    element.local_tangent(values, weights, source, zero, stiffness);

    REQUIRE(stiffness.rows() == rows);
    REQUIRE(stiffness.cols() == rows);

    // The form is symmetric, and the pressures do not meet
    REQUIRE((stiffness - stiffness.transpose()).cwiseAbs().maxCoeff()
            < 1e-12);
    REQUIRE(stiffness.bottomRightCorner(pressures, pressures).cwiseAbs()
                .maxCoeff() == 0.);

    // For linear velocities, the viscous term gives μ area (A : B + A : B^T)
    const double strains = a.cwiseProduct(b).sum()
                           + a.cwiseProduct(b.transpose()).sum();

    REQUIRE_THAT(linear(degrees, b).dot(stiffness * linear(degrees, a)),
                 WithinRel(viscosity * area * strains, 1e-12));

    // The pressure q = y tests −(q, div u), u = (x y, 0) having the
    // divergence y, and the velocity v = (0, y) the pressure, −(div v, q)
    const Eigen::VectorXd q = local(degrees, {}, {}, {{1., 0, 1}});

    REQUIRE_THAT(q.dot(stiffness * local(degrees, {{1., 1, 1}}, {}, {})),
                 WithinRel(-width * std::pow(height, 3) / 3., 1e-12));
    REQUIRE_THAT(local(degrees, {}, {{1., 0, 1}}, {}).dot(stiffness * q),
                 WithinRel(-width * height * height / 2., 1e-12));

    // The body force loads each component of the velocity by its integral,
    // and not the pressure, the load being the residual at zero negated
    Eigen::VectorXd load;
    element.local_residual(values, weights, source, zero, load);
    load = -load;

    REQUIRE(load.size() == rows);
    REQUIRE_THAT(local(degrees, {{1., 0, 0}}, {}, {}).dot(load),
                 WithinRel(.7 * area, 1e-12));
    REQUIRE_THAT(local(degrees, {}, {{1., 0, 0}}, {}).dot(load),
                 WithinRel(-1.2 * area, 1e-12));
    REQUIRE(load.tail(pressures).cwiseAbs().maxCoeff() == 0.);

    // The mass pairs each component of the velocity with itself, by the
    // integrals of x and of y, and not the pressure
    Eigen::MatrixXd mass;
    element.local_mass(values, weights, mass);

    REQUIRE(mass.rows() == rows);
    REQUIRE(mass.cols() == rows);
    REQUIRE_THAT(local(degrees, {{1., 0, 0}}, {}, {})
                     .dot(mass * local(degrees, {{1., 1, 0}}, {}, {})),
                 WithinRel(width * width * height / 2., 1e-12));
    REQUIRE_THAT(local(degrees, {}, {{1., 0, 0}}, {})
                     .dot(mass * local(degrees, {}, {{1., 0, 1}}, {})),
                 WithinRel(width * height * height / 2., 1e-12));
    REQUIRE(mass.bottomRows(pressures).cwiseAbs().maxCoeff() == 0.);
}

/// @brief Checks the terms the stabilization adds to the Galerkin forms
void check_stabilization(const Values& values, const Eigen::MatrixXd&,
                         const Eigen::VectorXd& weights,
                         const Degrees& degrees)
{
    const Element galerkin(viscosity);
    const Stabilized element(viscosity, stabilization);
    const double area = width * height;

    // τ_M = h^2 / (c μ), h^2 being the area of the element
    const double tau = area / (stabilization * viscosity);

    // A constant body force, and the tangents at zero coefficients, the
    // stiffnesses
    Eigen::MatrixXd source(weights.size(), 2);
    source.col(0).setConstant(.7);
    source.col(1).setConstant(-1.2);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(num_rows(degrees));

    Eigen::MatrixXd stiffness;
    Eigen::MatrixXd galerkin_stiffness;
    element.local_tangent(values, weights, source, zero, stiffness);
    galerkin.local_tangent(values, weights, source, zero, galerkin_stiffness);
    const Eigen::MatrixXd added = stiffness - galerkin_stiffness;

    // For linear velocities, grad-div gives μ area tr A tr B
    REQUIRE_THAT(linear(degrees, b).dot(added * linear(degrees, a)),
                 WithinRel(viscosity * area * a.trace() * b.trace(), 1e-12));

    // The gradient of q = y tests the momentum residual through
    // τ_M μ (∇q, Δu + ∇ div u) − τ_M (∇q, ∇p): u = (x y, 0) has the
    // divergence y and no Laplacian, and u = (0, x^2) no divergence and the
    // Laplacian (0, 2)
    const Eigen::VectorXd q = local(degrees, {}, {}, {{1., 0, 1}});

    REQUIRE_THAT(q.dot(added * local(degrees, {{1., 1, 1}}, {}, {})),
                 WithinRel(tau * viscosity * area, 1e-12));
    REQUIRE_THAT(q.dot(added * local(degrees, {}, {{1., 2, 0}}, {})),
                 WithinRel(2. * tau * viscosity * area, 1e-12));
    REQUIRE_THAT(q.dot(added * q), WithinRel(-tau * area, 1e-12));

    // The body force f loads the pressure by −τ_M (∇q, f), the load being
    // the residual at zero negated, and the mass pairs the pressure gradient
    // with the velocity, −τ_M (∇q, u)
    Eigen::VectorXd residual;
    Eigen::VectorXd galerkin_residual;
    element.local_residual(values, weights, source, zero, residual);
    galerkin.local_residual(values, weights, source, zero, galerkin_residual);

    REQUIRE_THAT(q.dot(galerkin_residual - residual),
                 WithinRel(1.2 * tau * area, 1e-12));

    Eigen::MatrixXd mass;
    Eigen::MatrixXd galerkin_mass;
    element.local_mass(values, weights, mass);
    galerkin.local_mass(values, weights, galerkin_mass);

    REQUIRE_THAT(q.dot((mass - galerkin_mass)
                       * local(degrees, {}, {{1., 0, 0}}, {})),
                 WithinRel(-tau * area, 1e-12));
}

/// @brief Checks that the traces and fluxes give the velocity A x, the
///        constant pressure p and the traction μ (A + A^T) n − p n
void check_traces(const Values& values, const Eigen::MatrixXd& points,
                  const Eigen::VectorXd&, const Degrees& degrees)
{
    const double pressure = .4;
    const Eigen::VectorXd fields =
        local(degrees, {{a(0, 0), 1, 0}, {a(0, 1), 0, 1}},
              {{a(1, 0), 1, 0}, {a(1, 1), 0, 1}}, {{pressure, 0, 0}});

    // The points in physical space, where the map scales each axis by its
    // side, and a normal at each
    const Eigen::Vector2d normal(.6, .8);
    const Eigen::MatrixXd positions =
        points * Eigen::Vector2d(width, height).asDiagonal();
    const iguana::PointMatrix<double, 2> normals =
        normal.transpose().replicate(points.rows(), 1);
    const Eigen::Vector2d traction =
        (viscosity * (a + a.transpose())
         - pressure * Eigen::Matrix2d::Identity())
        * normal;

    Eigen::MatrixXd trace;
    Eigen::MatrixXd flux;

    for (std::size_t c = 0; c < 2; ++c) {
        INFO("component " << c);
        const Eigen::Index component = static_cast<Eigen::Index>(c);

        Element::U(c).local_trace(values, trace);
        Element::T(viscosity, c).local_flux(values, normals, flux);

        REQUIRE(trace.rows() == fields.size());
        REQUIRE(flux.rows() == fields.size());
        REQUIRE((trace.transpose() * fields
                 - positions * a.row(component).transpose())
                    .cwiseAbs()
                    .maxCoeff() < 1e-12);
        REQUIRE(((flux.transpose() * fields).array() - traction(component))
                    .abs()
                    .maxCoeff() < 1e-12);
    }

    Element::P().local_trace(values, trace);

    REQUIRE(trace.rows() == fields.size());
    REQUIRE(((trace.transpose() * fields).array() - pressure).abs().maxCoeff()
            < 1e-12);
}

} // namespace

TEST_CASE("The Galerkin forms of a Stokes element are exact and symmetric "
          "in one field and in a field per unknown", "[stokes_element]")
{
    for_each_layout(check_galerkin);
}

TEST_CASE("A stabilized Stokes element adds the residuals of the momentum "
          "and continuity equations to the Galerkin forms",
          "[stokes_element]")
{
    for_each_layout(check_stabilization);
}

TEST_CASE("The traces and fluxes of a Stokes element read the field of "
          "each unknown", "[stokes_element]")
{
    for_each_layout(check_traces);

    // The unknowns share one field, or the pressure follows the velocity
    REQUIRE(Element::U(1).field(1) == 0);
    REQUIRE(Element::U(1).field(3) == 1);
    REQUIRE(Element::P().field(1) == 0);
    REQUIRE(Element::P().field(3) == 2);
}

TEST_CASE("A Stokes element needs a positive viscosity and stabilization, "
          "and its traces a component of the velocity", "[stokes_element]")
{
    REQUIRE_THROWS_AS(Element(0.), std::invalid_argument);
    REQUIRE_THROWS_AS(Stabilized(0., stabilization), std::invalid_argument);
    REQUIRE_THROWS_AS(Stabilized(viscosity, -1.), std::invalid_argument);
    REQUIRE_THROWS_AS(Element::U(2), std::invalid_argument);
    REQUIRE_THROWS_AS(Element::T(viscosity, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(Element::T(0., 0), std::invalid_argument);
}
