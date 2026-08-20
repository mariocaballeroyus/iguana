/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "iguana/assembly/assembler.hpp"

#include<cmath>
#include<cstddef>
#include<memory>
#include<vector>

#include<catch2/catch_test_macros.hpp>

namespace
{

using namespace iguana;

/// @brief A mass integrand with a unit source: the simplest physics
///        that exercises both scatters.
struct Mass final : Element<2, double>
{
    UpdateFlags updates() const noexcept override
    { return update_values | update_weights; }

    void residual(const ElementData<2, double>& data,
                  Eigen::VectorXd& residual) const override
    { residual -= data.values[0] * data.weights; }

    void jacobian(const ElementData<2, double>& data,
                  Eigen::MatrixXd& jacobian) const override
    {
        jacobian.noalias() += data.values[0]
                              * data.weights.asDiagonal()
                              * data.values[0].transpose();
    }
};

/// @brief A quadratic basis of the unit interval, four elements.
BSpline<double> quadratic()
{
    return BSpline<double>(2, {0., 0., 0., .25, .5, .75, 1., 1., 1.});
}

/// @brief The identity patch of the unit square at z = 0, its control
///        points on the Greville abscissae.
Patch<2, double> unit_square(const TensorProductBSpline<2, double>& basis)
{
    const BSpline<double> axis = quadratic();
    std::vector<double> greville;

    for (int i = 0; i < axis.num_functions(); ++i) {
        double sum = 0.;

        for (int j = 1; j <= axis.degree(); ++j)
            sum += axis.knots()[static_cast<std::size_t>(i + j)];

        greville.push_back(sum / axis.degree());
    }

    Eigen::MatrixX3<double> coefs(basis.num_functions(), 3);
    int row = 0;

    for (std::size_t j = 0; j < greville.size(); ++j)
        for (std::size_t i = 0; i < greville.size(); ++i, ++row)
            coefs.row(row) << greville[i], greville[j], 0.;

    return Patch<2, double>(basis, coefs);
}

} // namespace

TEST_CASE("The assembler scatters the local systems by the actives",
          "[assembly]")
{
    const TensorProductBSpline<2, double> basis(
        {quadratic(), quadratic()});

    const Assembler<2, double> assembler(TensorDomain<2, double>(basis),
                                         unit_square(basis),
                                         GaussLegendre<2, double>(3),
                                         std::make_unique<Mass>());

    Eigen::SparseMatrix<double> jacobian;
    Eigen::VectorXd residual;
    assembler.assemble(jacobian, residual);

    // Both scatters compute the integrals of the functions: the mass
    // against the partition of unity equals minus the load, entry by
    // entry, through two independent paths
    const Eigen::VectorXd ones
        = Eigen::VectorXd::Ones(basis.num_functions());

    REQUIRE((jacobian * ones + residual).cwiseAbs().maxCoeff() < 1e-13);

    // The mass is symmetric, and the total load is the area
    const Eigen::SparseMatrix<double> transposed = jacobian.transpose();

    REQUIRE((Eigen::MatrixXd(jacobian) - Eigen::MatrixXd(transposed))
                .cwiseAbs()
                .maxCoeff()
            < 1e-13);
    REQUIRE(std::abs(-residual.sum() - 1.) < 1e-13);
}
