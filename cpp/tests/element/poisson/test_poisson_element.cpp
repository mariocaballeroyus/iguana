/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <unsupported/Eigen/KroneckerProduct>

namespace
{

using Catch::Matchers::WithinAbs;
using Eigen::kroneckerProduct;
using iguana::BoxRule;
using Basis = iguana::TensorBSpline<double, 2>;

/// @brief Sides of the rectangle, unequal so that physical and parametric
///        gradients differ
constexpr double width = 2.;
constexpr double height = 3.;

/// @brief Rectangle [0, width] x [0, height] of degree 1 with one element,
///        a map that scales each axis by its side
iguana::Patch<Basis, 2> rectangle()
{
    const iguana::KnotVector<double> knots(1, {0., 0., 1., 1.});
    const Basis basis{iguana::TensorGrid<double, 2>({knots, knots})};

    iguana::PointMatrix<double, 2> points(4, 2);
    points << 0., 0.,
              width, 0.,
              0., height,
              width, height;

    return {basis, points};
}

} // namespace

TEST_CASE("A Poisson element integrates its forms on a rectangle exactly",
          "[poisson_element]")
{
    const iguana::Patch<Basis, 2> patch = rectangle();
    const iguana::PoissonElement<Basis, 2> element;
    iguana::ElementValues<Basis, 2> values(patch, element.flags());

    // Two points per direction integrate the products of linear functions
    const std::array<double, 2> start{0., 0.};
    const std::array<double, 2> end{1., 1.};
    const iguana::GaussLegendre<double, 2> rule(2);
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.fill_to_reference_space(start, end, points, weights);
    BoxRule<double, 2>::map_to_parameter_space(start, end, points, weights);
    values.reinit(0, points);

    // Physical weights, as an assembler gives them
    weights = weights.cwiseProduct(values.measures());

    // Stiffness and mass of a linear element on the unit interval. The
    // second direction is the outer factor, as the first runs fastest
    Eigen::Matrix2d s;
    s << 1., -1.,
         -1., 1.;
    Eigen::Matrix2d m;
    m << 1. / 3., 1. / 6.,
         1. / 6., 1. / 3.;

    const Eigen::Matrix4d expected = height / width * kroneckerProduct(m, s)
                                     + width / height * kroneckerProduct(s, m);

    // The source f = y, with y = height times the second parameter, and the
    // tangent at zero coefficients, the stiffness
    const Eigen::VectorXd source = height * points.col(1);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(4);

    Eigen::MatrixXd stiffness;
    element.local_tangent(values, weights, source, zero, stiffness);

    REQUIRE((stiffness - expected).cwiseAbs().maxCoeff() < 1e-14);

    // The mass is that of each direction, scaled by the area
    Eigen::MatrixXd mass;
    element.local_mass(values, weights, mass);

    REQUIRE((mass - width * height * kroneckerProduct(m, m)).cwiseAbs()
                .maxCoeff()
            < 1e-14);

    // The residual at zero coefficients is the load, negated. The functions
    // sum to one, so the load sums to the integral of y
    Eigen::VectorXd residual;
    element.local_residual(values, weights, source, zero, residual);

    REQUIRE_THAT(-residual.sum(),
                 WithinAbs(width * height * height / 2., 1e-14));
}
