/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;
using iguana::Boundary;
using iguana::PointMatrix;
using iguana::ShiftCenter;
using iguana::TensorBSpline;
using iguana::TensorNURBS;
using iguana::simplex_index;

/// @brief Ellipse of semi-axes a and b around a center, counterclockwise,
///        as four quadratic arcs joined at double knots
Boundary<double, 2>::Face ellipse(const Eigen::RowVector2d& center, double a,
                                  double b)
{
    const double corner = std::sqrt(.5);
    const TensorBSpline<double, 1> bspline({BSpline<double>(
        2, {0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.})});

    Eigen::VectorXd weights(9);
    weights << 1., corner, 1., corner, 1., corner, 1., corner, 1.;

    PointMatrix<double, 2> points(9, 2);
    points << 1., 0.,
              1., 1.,
              0., 1.,
             -1., 1.,
             -1., 0.,
             -1., -1.,
              0., -1.,
              1., -1.,
              1., 0.;
    points.col(0) *= a;
    points.col(1) *= b;
    points.rowwise() += center;

    return {TensorNURBS<double, 1>(bspline, weights), points};
}

/// @brief Patch of one bilinear element stretching the unit square onto a
///        box of the plane from the origin, with its sizes
iguana::Patch<TensorBSpline<double, 2>, 2> box(
    const Eigen::RowVector2d& sizes)
{
    const BSpline<double> unit(1, {0., 0., 1., 1.});

    PointMatrix<double, 2> corners(4, 2);
    corners << 0., 0.,
               1., 0.,
               0., 1.,
               1., 1.;
    corners = corners.array().rowwise() * sizes.array();

    return {TensorBSpline<double, 2>({unit, unit}), corners};
}

/// @brief Polynomial of a total degree with coefficients c_ij of x^i y^j
struct Polynomial
{
    int degree;
    Eigen::MatrixXd coefficients;

    /// @brief Its derivative d_x^a d_y^b at a point
    double derivative(int a, int b, const Eigen::RowVector2d& point) const
    {
        double sum = 0.;

        for (int i = a; i <= degree; ++i) {
            for (int j = b; i + j <= degree; ++j) {
                double factor = coefficients(i, j);

                for (int k = 0; k < a; ++k)
                    factor *= i - k;

                for (int k = 0; k < b; ++k)
                    factor *= j - k;

                sum += factor * std::pow(point(0), i - a)
                       * std::pow(point(1), j - b);
            }
        }

        return sum;
    }
};

/// @brief Coefficients of a total degree drawn at random
Eigen::MatrixXd random_coefficients(int degree, std::mt19937& generator)
{
    std::uniform_real_distribution<double> uniform(-1., 1.);
    Eigen::MatrixXd coefficients =
        Eigen::MatrixXd::Zero(degree + 1, degree + 1);

    for (int i = 0; i <= degree; ++i)
        for (int j = 0; i + j <= degree; ++j)
            coefficients(i, j) = uniform(generator);

    return coefficients;
}

} // namespace

TEST_CASE("The shift rewritten with the Stokes equation reproduces flows "
          "free of divergence at the surrogate points", "[stokes_shift]")
{
    // An ellipse inside a patch stretched by 2 and 3, whose parametric
    // derivatives are 2^a 3^b times the Cartesian ones
    const Eigen::RowVector2d sizes(2., 3.);
    const iguana::Patch<TensorBSpline<double, 2>, 2> patch = box(sizes);
    const Eigen::RowVector2d center(1., 1.4);
    const double a = .7;
    const double b = .45;
    const Boundary<double, 2> boundary({ellipse(center, a, b)}, {1});

    // Points inside and outside, around the ellipse, in parameters
    PointMatrix<double, 2> points(12, 2);
    Eigen::Index row = 0;

    for (const double scale : {.8, 1.2})
        for (const double angle : {.3, 1.1, 2., 3.3, 4.4, 5.6})
            points.row(row++) =
                (center + scale * Eigen::RowVector2d(a * std::cos(angle),
                                                     b * std::sin(angle)))
                    .cwiseQuotient(sizes);

    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(patch, boundary, points);
    const double viscosity = 1.7;
    std::mt19937 generator(3);

    for (const ShiftCenter center :
         {ShiftCenter::boundary, ShiftCenter::surrogate}) {
        const bool from_boundary = center == ShiftCenter::boundary;

        for (int m = 1; m <= 6; ++m) {
            INFO("from the boundary " << from_boundary << ", order " << m);

            // u = (∂_y ψ, −∂_x ψ) of a stream function ψ of degree m + 1, and
            // a pressure of degree m − 1
            const Eigen::MatrixXd psi = random_coefficients(m + 1, generator);
            Eigen::MatrixXd first = Eigen::MatrixXd::Zero(m + 1, m + 1);
            Eigen::MatrixXd second = Eigen::MatrixXd::Zero(m + 1, m + 1);

            for (int i = 0; i <= m + 1; ++i) {
                for (int j = 0; i + j <= m + 1; ++j) {
                    if (j > 0)
                        first(i, j - 1) += j * psi(i, j);

                    if (i > 0)
                        second(i - 1, j) -= i * psi(i, j);
                }
            }

            const Polynomial u[2] = {{m, first}, {m, second}};
            const Polynomial p{m - 1, random_coefficients(m - 1, generator)};
            const iguana::StokesShift<double> shift = iguana::stokes_shift(
                patch, projection, m, viscosity, center);

            for (Eigen::Index point = 0; point < points.rows(); ++point) {
                const Eigen::RowVector2d surrogate =
                    points.row(point).cwiseProduct(sizes);
                const Eigen::RowVector2d closest =
                    projection.positions.row(point);
                const Eigen::RowVector2d at =
                    from_boundary ? closest : surrogate;

                for (int c = 0; c < 2; ++c) {
                    INFO("point " << point << ", component " << c);

                    // From the boundary, u_c at the point and the
                    // parametric derivatives of u_c and of p at its closest
                    // point; from the surrogate, the Taylor series of u_c,
                    // exact, and the derivatives at the point. Against g = u
                    // at the closest point and f = −μ Δu + ∇p where the
                    // derivatives are taken
                    double shifted = u[c].derivative(
                        0, 0, from_boundary ? surrogate : closest);

                    for (int q = 0; q <= m; ++q) {
                        for (int s = 0; s + q <= m; ++s) {
                            const double scale =
                                std::pow(2., s) * std::pow(3., q);
                            const int column = simplex_index(s, q, m);

                            shifted += scale
                                       * (shift.weights(point, column)
                                          * u[c].derivative(s, q, at)
                                          + shift.pressure_weights[c](point,
                                                                      column)
                                                * p.derivative(s, q, at));
                        }
                    }

                    double data = u[c].derivative(0, 0, closest);

                    for (int q = 0; q + 2 <= m; ++q) {
                        for (int s = 0; s + q + 2 <= m; ++s) {
                            const double force =
                                -viscosity * (u[c].derivative(s + 2, q, at)
                                              + u[c].derivative(s, q + 2, at))
                                + p.derivative(s + (c == 0 ? 1 : 0),
                                               q + (c == 1 ? 1 : 0), at);

                            data += shift.source_weights(
                                        point, simplex_index(s, q, m - 2))
                                    * force;
                        }
                    }

                    REQUIRE(std::abs(shifted - data) < 1e-11);
                }
            }
        }
    }

    // A viscosity that is not positive, a negative order and a map that is
    // not affine
    REQUIRE_THROWS_AS(iguana::stokes_shift(patch, projection, 2, 0.),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::stokes_shift(patch, projection, -1, 1.),
                      std::invalid_argument);

    PointMatrix<double, 2> bent = patch.coefficients();
    bent(3, 0) += .5;
    const iguana::Patch<TensorBSpline<double, 2>, 2> curved(patch.basis(),
                                                            bent);

    REQUIRE_THROWS_AS(iguana::stokes_shift(curved, projection, 2, 1.),
                      std::invalid_argument);
}
