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

/// @brief Weights of v · ∇u
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

TEST_CASE("Derivatives along a frame combine the partial derivatives along "
          "its directions", "[directional_derivs]")
{
    // A frame that is not orthonormal, as one mapped into parameters is
    const Eigen::Vector2d f0(.8, .3);
    const Eigen::Vector2d f1(-.2, 1.1);
    Eigen::Matrix2d frame;
    frame << f0, f1;

    const Eigen::MatrixXd derivs = iguana::directional_derivs(frame, order);
    const auto row = [&](int j, int i) {
        return derivs.row(simplex_index(j, i, order));
    };

    REQUIRE(derivs.rows() == simplex_size(order));
    REQUIRE(derivs.cols() == simplex_size(order));

    REQUIRE((row(0, 0).transpose()
             - Eigen::VectorXd::Unit(simplex_size(order), 0)).norm() == 0.);
    REQUIRE((row(1, 0) - gradient(f0)).norm() < 1e-15);
    REQUIRE((row(0, 1) - gradient(f1)).norm() < 1e-15);
    REQUIRE((row(2, 0) - hessian(f0, f0)).norm() < 1e-15);
    REQUIRE((row(1, 1) - hessian(f0, f1)).norm() < 1e-15);
    REQUIRE((row(2, 1) - third(f0, f1)).norm() < 1e-15);
    REQUIRE((row(1, 2) - third(f1, f0)).norm() < 1e-15);

    REQUIRE_THROWS_AS(iguana::directional_derivs(frame, -1),
                      std::invalid_argument);
}
