/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <stdexcept>

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
using Element = iguana::StokesStabilizedElement<Basis, 2>;

/// @brief Sides of the rectangle, unequal so that physical and parametric
///        derivatives differ
constexpr double width = 2.;
constexpr double height = 3.;

/// @brief Dynamic viscosity and stabilization constant of the element
constexpr double viscosity = 1.5;
constexpr double stabilization = 3.;

/// @brief Rectangle [0, width] x [0, height] of degree 2 with one element,
///        whose control points are the scaled Greville points, so that the
///        map scales each axis by its side
iguana::Patch<Basis, 2> rectangle()
{
    const iguana::KnotVector<double> knots(2, {0., 0., 0., 1., 1., 1.});
    const Basis basis{iguana::TensorGrid<double, 2>({knots, knots})};

    // The first direction runs fastest
    iguana::PointMatrix<double, 2> points(9, 2);

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i)
            points.row(i + 3 * j) << width * i / 2., height * j / 2.;
    }

    return {basis, points};
}

/// @brief Coefficients of a product of a polynomial along x and one along
///        y, given by their coefficients along each, the first direction
///        running fastest
Eigen::VectorXd product(const Eigen::Vector3d& along_x,
                        const Eigen::Vector3d& along_y)
{
    return kroneckerProduct(along_y, along_x);
}

/// @brief Local coefficients of the velocity and the pressure, one
///        component after another
Eigen::VectorXd local(const Eigen::VectorXd& u_x, const Eigen::VectorXd& u_y,
                      const Eigen::VectorXd& p)
{
    Eigen::VectorXd result(27);
    result << u_x, u_y, p;

    return result;
}

} // namespace

TEST_CASE("A Stokes element integrates its forms on a rectangle exactly",
          "[stokes_element]")
{
    const iguana::Patch<Basis, 2> patch = rectangle();
    const Element element(viscosity, stabilization);
    iguana::ElementValues<Basis, 2> values(patch, element.flags());

    // Three points per direction integrate the products of quadratic
    // functions
    const std::array<double, 2> start{0., 0.};
    const std::array<double, 2> end{1., 1.};
    const iguana::GaussLegendre<double, 2> rule(3);
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.fill_to_reference_space(start, end, points, weights);
    BoxRule<double, 2>::map_to_parameter_space(start, end, points, weights);
    values.reinit(0, points);

    // Physical weights, as an assembler gives them
    weights = weights.cwiseProduct(values.measures());

    // The coefficients of 1, t and t^2 in the quadratic Bernstein basis,
    // their blossoms, give those of the fields
    const Eigen::Vector3d one(1., 1., 1.);
    const Eigen::Vector3d t(0., .5, 1.);
    const Eigen::Vector3d t_squared(0., 0., 1.);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(9);
    const Eigen::VectorXd constant = product(one, one);
    const Eigen::VectorXd x = product(width * t, one);
    const Eigen::VectorXd y = product(one, height * t);
    const Eigen::VectorXd xy = product(width * t, height * t);
    const Eigen::VectorXd x_squared = product(width * width * t_squared, one);

    // τ_M = h^2 / (c μ), h^2 being the area of the element
    const double area = width * height;
    const double tau = area / (stabilization * viscosity);

    Eigen::MatrixXd stiffness;
    element.local_stiffness(values, weights, stiffness);

    // For linear velocities u = A x and v = B x, the viscous and grad-div
    // terms give μ area (A : B + A : B^T + tr A tr B)
    Eigen::Matrix2d a;
    a << 1., -2., 3., .5;
    Eigen::Matrix2d b;
    b << -1., 2., .5, 4.;

    const Eigen::VectorXd u =
        local(a(0, 0) * x + a(0, 1) * y, a(1, 0) * x + a(1, 1) * y, zero);
    const Eigen::VectorXd v =
        local(b(0, 0) * x + b(0, 1) * y, b(1, 0) * x + b(1, 1) * y, zero);
    const double strains = a.cwiseProduct(b).sum()
                           + a.cwiseProduct(b.transpose()).sum()
                           + a.trace() * b.trace();

    REQUIRE_THAT(v.dot(stiffness * u),
                 WithinRel(viscosity * area * strains, 1e-12));

    // The pressure q = y tests −(q, div u) and the momentum residual through
    // τ_M μ (∇q, Δu + ∇ div u): u = (x y, 0) has the divergence y and no
    // Laplacian, and u = (0, x^2) no divergence and the Laplacian (0, 2)
    const Eigen::VectorXd q = local(zero, zero, y);

    REQUIRE_THAT(q.dot(stiffness * local(xy, zero, zero)),
                 WithinRel(-width * std::pow(height, 3) / 3.
                               + tau * viscosity * area,
                           1e-12));
    REQUIRE_THAT(q.dot(stiffness * local(zero, x_squared, zero)),
                 WithinRel(2. * tau * viscosity * area, 1e-12));

    // The pressure gradient tests the momentum residual, −τ_M (∇q, ∇p), and
    // the velocity v = (0, y) the pressure, −(div v, p)
    REQUIRE_THAT(q.dot(stiffness * q), WithinRel(-tau * area, 1e-12));
    REQUIRE_THAT(local(zero, y, zero).dot(stiffness * q),
                 WithinRel(-width * height * height / 2., 1e-12));

    // A constant body force f loads each component of the velocity by its
    // integral, and the pressure by −τ_M (∇q, f)
    Eigen::MatrixXd source(points.rows(), 2);
    source.col(0).setConstant(.7);
    source.col(1).setConstant(-1.2);

    Eigen::VectorXd load;
    element.local_load(values, weights, source, load);

    REQUIRE_THAT(local(constant, zero, zero).dot(load),
                 WithinRel(.7 * area, 1e-12));
    REQUIRE_THAT(q.dot(load), WithinRel(1.2 * tau * area, 1e-12));

    // The mass pairs the velocities, (1, 0) and (x, 0) by the integral of x,
    // and the pressure gradient with the velocity, −τ_M (∇q, u)
    Eigen::MatrixXd mass;
    element.local_mass(values, weights, mass);

    REQUIRE_THAT(local(constant, zero, zero).dot(mass * local(x, zero, zero)),
                 WithinRel(width * width * height / 2., 1e-12));
    REQUIRE_THAT(q.dot(mass * local(zero, constant, zero)),
                 WithinRel(-tau * area, 1e-12));
}

TEST_CASE("A Stokes element needs a positive viscosity and stabilization, "
          "and its traces a component of the velocity", "[stokes_element]")
{
    REQUIRE_THROWS_AS(Element(0., stabilization), std::invalid_argument);
    REQUIRE_THROWS_AS(Element(viscosity, -1.), std::invalid_argument);
    REQUIRE_THROWS_AS(Element::U(2), std::invalid_argument);
    REQUIRE_THROWS_AS(Element::T(viscosity, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(Element::T(0., 0), std::invalid_argument);
}
