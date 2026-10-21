/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::simplex_index;
using iguana::simplex_size;

constexpr int order = 3;

/// @brief Weights of v . grad u
Eigen::RowVectorXd gradient(const Eigen::Vector2d& v)
{
    Eigen::RowVectorXd weights = Eigen::RowVectorXd::Zero(simplex_size(order));
    weights(simplex_index(1, 0, order)) = v(0);
    weights(simplex_index(0, 1, order)) = v(1);
    return weights;
}

/// @brief Weights of v^T H w, with H the Hessian of u
Eigen::RowVectorXd hessian(const Eigen::Vector2d& v, const Eigen::Vector2d& w)
{
    Eigen::RowVectorXd weights = Eigen::RowVectorXd::Zero(simplex_size(order));
    weights(simplex_index(2, 0, order)) = v(0) * w(0);
    weights(simplex_index(1, 1, order)) = v(0) * w(1) + v(1) * w(0);
    weights(simplex_index(0, 2, order)) = v(1) * w(1);
    return weights;
}

/// @brief Weights of the third derivatives of u along v, v and w
Eigen::RowVectorXd third(const Eigen::Vector2d& v, const Eigen::Vector2d& w)
{
    Eigen::RowVectorXd weights = Eigen::RowVectorXd::Zero(simplex_size(order));
    weights(simplex_index(3, 0, order)) = v(0) * v(0) * w(0);
    weights(simplex_index(2, 1, order)) =
        v(0) * v(0) * w(1) + 2. * v(0) * v(1) * w(0);
    weights(simplex_index(1, 2, order)) =
        2. * v(0) * v(1) * w(1) + v(1) * v(1) * w(0);
    weights(simplex_index(0, 3, order)) = v(1) * v(1) * w(1);
    return weights;
}

} // namespace

TEST_CASE("Derivatives in normal coordinates follow from the Frenet "
          "relations", "[normal_coordinates]")
{
    // A frame that is not orthonormal, as one mapped into parameters is
    const Eigen::Vector2d tau(.8, .3);
    const Eigen::Vector2d nu(-.2, 1.1);
    Eigen::Matrix2d frame;
    frame << tau, nu;

    const double kappa = .7;
    const double kappa_prime = -.4;
    Eigen::VectorXd curvatures(2);
    curvatures << kappa, kappa_prime;

    const Eigen::MatrixXd derivs =
        iguana::normal_coordinate_derivs(frame, curvatures, order);

    REQUIRE(derivs.rows() == simplex_size(order));
    REQUIRE(derivs.cols() == simplex_size(order));

    // d²u/ds² = τ^T H τ − κ ν · ∇u, d/ds (ν · ∇u) = κ τ · ∇u + τ^T H ν, and
    // d²/ds² (ν · ∇u) = κ' τ · ∇u − κ² ν · ∇u + 2 κ τ^T H τ − κ ν^T H ν
    //                   + ∇³u[τ, τ, ν]
    const Eigen::RowVectorXd along = hessian(tau, tau) - kappa * gradient(nu);
    const Eigen::RowVectorXd across =
        kappa * gradient(tau) + hessian(tau, nu);
    const Eigen::RowVectorXd twice =
        kappa_prime * gradient(tau) - kappa * kappa * gradient(nu)
        + 2. * kappa * hessian(tau, tau) - kappa * hessian(nu, nu)
        + third(tau, nu);

    REQUIRE((derivs.row(simplex_index(2, 0, order)) - along).norm() < 1e-15);
    REQUIRE((derivs.row(simplex_index(1, 1, order)) - across).norm()
            < 1e-15);
    REQUIRE((derivs.row(simplex_index(2, 1, order)) - twice).norm()
            < 1e-15);
}

TEST_CASE("Derivatives in the normal coordinates of a line are those along "
          "its frame", "[normal_coordinates]")
{
    // A rotated frame of a straight line, so that every row is a product of
    // derivatives along τ and ν
    const Eigen::Vector2d tau(.6, .8);
    const Eigen::Vector2d nu(.8, -.6);
    Eigen::Matrix2d frame;
    frame << tau, nu;

    const Eigen::VectorXd straight = Eigen::VectorXd::Zero(2);
    const Eigen::MatrixXd derivs =
        iguana::normal_coordinate_derivs(frame, straight, order);

    REQUIRE((derivs.row(simplex_index(0, 0, order)).transpose()
             - Eigen::VectorXd::Unit(simplex_size(order), 0)).norm() == 0.);
    REQUIRE((derivs.row(simplex_index(0, 1, order)) - gradient(nu)).norm()
            < 1e-15);
    REQUIRE((derivs.row(simplex_index(2, 0, order)) - hessian(tau, tau))
                .norm() < 1e-15);
    REQUIRE((derivs.row(simplex_index(2, 1, order)) - third(tau, nu)).norm()
            < 1e-15);
    REQUIRE((derivs.row(simplex_index(1, 1, order)) - hessian(tau, nu))
                .norm() < 1e-15);

    // A negative order, and one curvature short for the order
    REQUIRE_THROWS_AS(iguana::normal_coordinate_derivs(frame, straight, -1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(iguana::normal_coordinate_derivs(
                          frame, Eigen::VectorXd(straight.head(1)), 3),
                      std::invalid_argument);
}
