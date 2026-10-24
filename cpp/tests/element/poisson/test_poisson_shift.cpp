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
using iguana::TensorBSpline;
using iguana::TensorNURBS;
using iguana::simplex_index;
using iguana::simplex_size;

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

/// @brief Straight face from one point to another, a line of degree one
Boundary<double, 2>::Face segment(const Eigen::RowVector2d& from,
                                  const Eigen::RowVector2d& to)
{
    PointMatrix<double, 2> points(2, 2);
    points << from, to;

    const TensorBSpline<double, 1> line(
        {BSpline<double>(1, {0., 0., 1., 1.})});

    return {TensorNURBS<double, 1>(line, Eigen::Vector2d::Ones()), points};
}

/// @brief Patch of one bilinear element stretching the unit square onto a
///        box of the plane, from its lowest corner and with its sizes
iguana::Patch<TensorBSpline<double, 2>, 2> box(
    const Eigen::RowVector2d& corner, const Eigen::RowVector2d& sizes)
{
    const BSpline<double> unit(1, {0., 0., 1., 1.});

    PointMatrix<double, 2> corners(4, 2);
    corners << 0., 0.,
               1., 0.,
               0., 1.,
               1., 1.;
    corners = corners.array().rowwise() * sizes.array();
    corners.rowwise() += corner;

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

/// @brief A polynomial of a total degree with random coefficients
Polynomial random_polynomial(int degree, std::mt19937& generator)
{
    std::uniform_real_distribution<double> uniform(-1., 1.);
    Eigen::MatrixXd coefficients =
        Eigen::MatrixXd::Zero(degree + 1, degree + 1);

    for (int i = 0; i <= degree; ++i)
        for (int j = 0; i + j <= degree; ++j)
            coefficients(i, j) = uniform(generator);

    return {degree, coefficients};
}

} // namespace

TEST_CASE("The shift rewritten with the Poisson equation reproduces "
          "polynomials at the surrogate points, off corners too",
          "[poisson_shift]")
{
    // A patch stretched by 2 and 3, whose parametric derivatives are
    // 2^a 3^b times the Cartesian ones
    const Eigen::RowVector2d sizes(2., 3.);
    const iguana::Patch<TensorBSpline<double, 2>, 2> patch =
        box({0., 0.}, sizes);
    std::mt19937 generator(7);

    // u at each point and the parametric derivatives at its closest point,
    // against g = u and f = −Δu there, for every order
    const auto reproduces = [&](const char* shape,
                                const Boundary<double, 2>& boundary,
                                const PointMatrix<double, 2>& positions) {
        INFO(shape);
        const PointMatrix<double, 2> points =
            positions.array().rowwise() / sizes.array();
        const iguana::BoundaryProjection<double> projection =
            iguana::project_closest_points(patch, boundary, points);

        for (int m = 1; m <= 6; ++m) {
            INFO("order " << m);
            const Polynomial u = random_polynomial(m, generator);
            const iguana::PoissonShift<double> shift =
                iguana::poisson_shift(patch, projection, m);

            REQUIRE(shift.weights.cols() == simplex_size(m));

            for (Eigen::Index point = 0; point < points.rows(); ++point) {
                INFO("point " << point);
                const Eigen::RowVector2d surrogate = positions.row(point);
                const Eigen::RowVector2d closest =
                    projection.positions.row(point);

                double shifted = u.derivative(0, 0, surrogate);

                for (int q = 0; q <= m; ++q)
                    for (int p = 0; p + q <= m; ++p)
                        shifted += shift.weights(point, simplex_index(p, q, m))
                                   * std::pow(2., p) * std::pow(3., q)
                                   * u.derivative(p, q, closest);

                double data = u.derivative(0, 0, closest);

                for (int q = 0; q + 2 <= m; ++q)
                    for (int p = 0; p + q + 2 <= m; ++p)
                        data -= shift.source_weights(
                                    point, simplex_index(p, q, m - 2))
                                * (u.derivative(p + 2, q, closest)
                                   + u.derivative(p, q + 2, closest));

                REQUIRE(std::abs(shifted - data) < 1e-11);
            }
        }
    };

    // Points inside and outside an ellipse, around it
    const Eigen::RowVector2d center(1., 1.4);
    const double a = .7;
    const double b = .45;
    PointMatrix<double, 2> around(12, 2);
    Eigen::Index row = 0;

    for (const double scale : {.8, 1.2})
        for (const double angle : {.3, 1.1, 2., 3.3, 4.4, 5.6})
            around.row(row++) =
                center + scale * Eigen::RowVector2d(a * std::cos(angle),
                                                    b * std::sin(angle));

    reproduces("ellipse", Boundary<double, 2>({ellipse(center, a, b)}, {1}),
               around);

    // Points beside the sides of a square, inside and outside, and the last
    // four off its corners, whose closest points are the vertices
    const Boundary<double, 2> square(
        {segment({.5, 1.}, {1.5, 1.}), segment({1.5, 1.}, {1.5, 2.}),
         segment({1.5, 2.}, {.5, 2.}), segment({.5, 2.}, {.5, 1.})},
        {1, 1, 1, 1});

    PointMatrix<double, 2> beside(8, 2);
    beside << 1., 1.2,
              1.3, 1.5,
              1., .8,
              .3, 1.6,
              1.7, 2.3,
              .2, .7,
              1.8, .6,
              .3, 2.4;

    reproduces("square", square, beside);
}

TEST_CASE("The shift rewritten with the equation needs an affine map and "
          "a non-negative order", "[poisson_shift]")
{
    const iguana::Patch<TensorBSpline<double, 2>, 2> patch =
        box({0., 0.}, {1., 1.});
    const iguana::BoundaryProjection<double> projection;

    REQUIRE_NOTHROW(iguana::poisson_shift(patch, projection, 2));
    REQUIRE_THROWS_AS(iguana::poisson_shift(patch, projection, -1),
                      std::invalid_argument);

    PointMatrix<double, 2> bent = patch.coefficients();
    bent(3, 0) += .5;
    const iguana::Patch<TensorBSpline<double, 2>, 2> curved(patch.basis(),
                                                            bent);

    REQUIRE_THROWS_AS(iguana::poisson_shift(curved, projection, 2),
                      std::invalid_argument);
}
