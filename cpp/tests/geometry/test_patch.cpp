/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "iguana/geometry/patch.hpp"

#include<array>
#include<cmath>
#include<cstddef>
#include<stdexcept>
#include<vector>

#include<catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;
using iguana::TensorProductBSpline;

using Patch2d = iguana::Patch<2, double>;

/// @brief The Greville abscissae, the knot averages at which control
///        points reproduce an affine map exactly.
std::vector<double> greville(const BSpline<double>& basis)
{
    std::vector<double> abscissae;

    for (int i = 0; i < basis.num_functions(); ++i) {
        if (basis.degree() == 0) {
            abscissae.push_back(basis.knots()[static_cast<std::size_t>(i)]);
            continue;
        }

        double sum = 0.;

        for (int j = 1; j <= basis.degree(); ++j)
            sum += basis.knots()[static_cast<std::size_t>(i + j)];

        abscissae.push_back(sum / basis.degree());
    }

    return abscissae;
}

/// @brief Points spread inside an element of a bivariate basis.
Eigen::MatrixXd points_on(const TensorProductBSpline<2, double>& basis,
                          int element)
{
    const std::array<double, 2> start = basis.element_start(element);
    const std::array<double, 2> end = basis.element_end(element);

    Eigen::MatrixXd points(3, 2);

    for (int k = 0; k < 2; ++k)
        for (int q = 0; q < 3; ++q)
            points(q, k) = start[k]
                           + (end[k] - start[k]) * (.25 + .25 * q);

    return points;
}

} // namespace

TEST_CASE("A patch rejects control points that do not match the basis",
          "[geometry]")
{
    const TensorProductBSpline<2, double> basis(
        {BSpline<double>(2, {0., 0., 0., 1., 1., 1.}),
         BSpline<double>(1, {0., 0., 1., 1.})});

    const Eigen::MatrixX3<double> wrong
        = Eigen::MatrixX3<double>::Zero(basis.num_functions() + 1, 3);

    REQUIRE_THROWS_AS(Patch2d(basis, wrong), std::invalid_argument);
}

TEST_CASE("A patch over Greville points reproduces an affine map",
          "[geometry]")
{
    const BSpline<double> bu(2, {0., 0., 0., 1., 2., 3., 3., 3.});
    const BSpline<double> bv(3, {0., 0., 0., 0., 1., 2., 2., 2., 2.});
    const TensorProductBSpline<2, double> basis({bu, bv});

    // The map x = A u + c, whose tangents are the columns of A
    const double A[3][2] = {{2., .5}, {-1., 3.}, {.25, -.75}};

    const std::vector<double> gu = greville(bu);
    const std::vector<double> gv = greville(bv);

    Eigen::MatrixX3<double> coefs(basis.num_functions(), 3);
    int row = 0;

    for (std::size_t j = 0; j < gv.size(); ++j)
        for (std::size_t i = 0; i < gu.size(); ++i, ++row)
            for (int c = 0; c < 3; ++c)
                coefs(row, c) = A[c][0] * gu[i] + A[c][1] * gv[j] + 1.;

    const Patch2d patch(basis, coefs);

    Eigen::VectorXi actives;
    std::vector<Eigen::MatrixXd> ders;
    Eigen::MatrixX3<double> positions;
    std::array<Eigen::MatrixX3<double>, 2> tangents;

    for (int e = 0; e < basis.num_elements(); ++e) {
        const Eigen::MatrixXd points = points_on(basis, e);

        basis.active_on_element(e, actives);
        basis.eval_ders_on_element(basis.first_active(e), points, 1,
                                   ders);

        patch.position(actives, ders[0], positions);
        patch.tangents(actives, ders[1], tangents);

        for (int q = 0; q < 3; ++q)
            for (int c = 0; c < 3; ++c) {
                const double exact = A[c][0] * points(q, 0)
                                     + A[c][1] * points(q, 1) + 1.;

                REQUIRE(std::abs(positions(q, c) - exact) < 1e-13);

                for (int k = 0; k < 2; ++k)
                    REQUIRE(std::abs(tangents[
                                         static_cast<std::size_t>(k)](
                                         q, c)
                                     - A[c][k]) < 1e-13);
            }
    }
}
