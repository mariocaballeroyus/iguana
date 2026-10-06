/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;
using iguana::CellType;
using iguana::EmbeddedDomain;
using iguana::KnotVector;
using iguana::TensorGrid;
using Basis = iguana::TensorBSpline<double, 2>;

/// @brief Sides of the rectangle, so that the measure of the map is not one
constexpr double width = 2.;
constexpr double height = 3.;

/// @brief Quadratic basis over uneven knot spans of the unit square, with
///        three elements along the first direction and two along the second
Basis uneven()
{
    return Basis(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., .2, .6, 1., 1., 1.}),
         KnotVector<double>(2, {0., 0., 0., .7, 1., 1., 1.})}));
}

/// @brief Greville abscissa of a quadratic B-spline, (t_{i+1} + t_{i+2}) / 2
double greville(const KnotVector<double>& knot_vector, int function)
{
    const std::vector<double>& t = knot_vector.values();

    return (t[function + 1] + t[function + 2]) / 2.;
}

/// @brief Rectangle [0, width] x [0, height] over a basis, whose control
///        points are the scaled Greville points, so that the map scales
///        each axis by its side
iguana::Patch<Basis, 2> rectangle(const Basis& basis)
{
    iguana::PointMatrix<double, 2> points(basis.num_functions(), 2);
    const int count = basis.axis(0).num_functions();

    for (int function = 0; function < basis.num_functions(); ++function) {
        const int along_x = function % count;
        const int along_y = function / count;

        points(function, 0) = width * greville(basis.axis(0).knots(), along_x);
        points(function, 1) = height * greville(basis.axis(1).knots(), along_y);
    }

    return {basis, points};
}

/// @brief Inside cells, except those of the last column along the first
///        direction, which take the given type
EmbeddedDomain<double, 2> with_last_column(CellType type)
{
    return EmbeddedDomain<double, 2>(
        {CellType::inside, CellType::inside, type,
         CellType::inside, CellType::inside, type});
}

/// @brief Gauss quadrature of the inside cells, exact for the products of
///        quadratic functions through an affine map
iguana::DomainQuadrature<double, 2> inside_quadrature(
    const Basis& basis, const EmbeddedDomain<double, 2>& domain)
{
    iguana::DomainQuadrature<double, 2> quadrature;
    quadrature.fill(basis.grid(), domain, CellType::inside,
                    iguana::GaussLegendre<double, 2>(3));

    return quadrature;
}

} // namespace

TEST_CASE("The Poisson system of a rectangle integrates exactly",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const EmbeddedDomain<double, 2> domain = with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, domain);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, domain);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);

    // The source f = y, with y = height times the second parameter
    const Eigen::VectorXd source = height * quadrature.points().col(1);

    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, source);

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& stiffness =
        assembler.stiffness();

    // With every cell inside, each function keeps its own index, and the
    // coefficients of b . x are b . x_i at the control points
    const Eigen::Vector2d b(1., -2.);
    const Eigen::VectorXd u = patch.coefficients() * b;
    const double area = width * height;

    REQUIRE_THAT(u.dot(stiffness * u),
                 WithinRel(b.squaredNorm() * area, 1e-12));

    // The functions sum to one, so constants have no energy and the load
    // sums to the integral of y
    const Eigen::VectorXd constant =
        Eigen::VectorXd::Ones(space.dof_map().num_dofs());

    REQUIRE((stiffness * constant).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE_THAT(assembler.load().sum(),
                 WithinRel(width * height * height / 2., 1e-12));
}

TEST_CASE("The Poisson system leaves out the functions of outside cells",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const EmbeddedDomain<double, 2> domain =
        with_last_column(CellType::outside);
    const iguana::FunctionSpace<Basis> space(basis, domain);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, domain);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(quadrature.num_points());

    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, ones);

    // The degrees of freedom number the functions compactly, and those
    // left sum to one over the inside cells, the parameters [0, .6] x [0, 1]
    const Eigen::VectorXd constant =
        Eigen::VectorXd::Ones(space.dof_map().num_dofs());

    REQUIRE((assembler.stiffness() * constant).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE_THAT(assembler.load().sum(),
                 WithinRel(.6 * width * height, 1e-12));
}

TEST_CASE("A cleared assembler assembles the same system again",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const EmbeddedDomain<double, 2> domain = with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, domain);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, domain);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(quadrature.num_points());

    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, ones);

    const Eigen::MatrixXd stiffness(assembler.stiffness());
    const Eigen::VectorXd load = assembler.load();

    // Assembling again after clearing gives the same system, not twice it
    assembler.clear();
    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, ones);

    const Eigen::MatrixXd again(assembler.stiffness());

    REQUIRE((again - stiffness).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE((assembler.load() - load).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("An assembler needs one source value per point", "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const EmbeddedDomain<double, 2> domain = with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, domain);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, domain);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);
    // One value short of the points
    const Eigen::VectorXd source =
        Eigen::VectorXd::Ones(quadrature.num_points() - 1);

    REQUIRE_THROWS_AS(assembler.assemble_load(element, quadrature, source),
                      std::invalid_argument);
}
