/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
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

/// @brief Straight face of a boundary from one point to another
iguana::Boundary<double, 2>::Face segment(const Eigen::RowVector2d& from,
                                          const Eigen::RowVector2d& to)
{
    iguana::PointMatrix<double, 2> points(2, 2);
    points << from, to;

    const iguana::TensorBSpline<double, 1> line(
        {iguana::BSpline<double>(1, {0., 0., 1., 1.})});

    return {iguana::TensorNURBS<double, 1>(line, Eigen::Vector2d::Ones()),
            points};
}

/// @brief Corners of a tilted quadrilateral inside the rectangle,
///        counterclockwise, whose sides cross the knot lines
std::array<Eigen::RowVector2d, 4> quadrilateral()
{
    return {Eigen::RowVector2d(.3, .4), Eigen::RowVector2d(1.7, .6),
            Eigen::RowVector2d(1.5, 2.6), Eigen::RowVector2d(.4, 2.2)};
}

/// @brief Quadrature of the boundary of a quadrilateral in a patch, with
///        five points per piece, as biquadratic functions are of degree four
///        along a line and their products integrate exactly
iguana::BoundaryQuadrature<double, 2> boundary_quadrature(
    const iguana::Patch<Basis, 2>& patch,
    const std::array<Eigen::RowVector2d, 4>& corners)
{
    std::vector<iguana::Boundary<double, 2>::Face> faces;

    for (std::size_t side = 0; side < 4; ++side)
        faces.push_back(segment(corners[side], corners[(side + 1) % 4]));

    const iguana::EmbeddedBoundary<double, 2> embedded(
        patch, iguana::Boundary<double, 2>(faces, {1, 1, 1, 1}));

    return {embedded, iguana::GaussLegendre<double, 1>(5)};
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

TEST_CASE("A penalty on an embedded boundary imposes a field of the space",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(
        basis, with_last_column(CellType::inside));

    const std::array<Eigen::RowVector2d, 4> corners = quadrilateral();
    const iguana::BoundaryQuadrature<double, 2> quadrature =
        boundary_quadrature(patch, corners);
    double perimeter = 0.;

    for (std::size_t side = 0; side < 4; ++side)
        perimeter += (corners[(side + 1) % 4] - corners[side]).norm();

    const iguana::PoissonElement<Basis, 2>::U trace;
    const double penalty = 10.;
    const iguana::PenaltyCondition<Basis, 2> condition(trace, penalty);

    // The data of b . x, the points lying in parameter space, where the map
    // scales each axis by its side
    const Eigen::Vector2d b(1., -2.);
    const Eigen::VectorXd data =
        width * b(0) * quadrature.points().col(0)
        + height * b(1) * quadrature.points().col(1);

    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(condition, quadrature);
    assembler.assemble_load(condition, quadrature, data);

    // The functions sum to one, so the penalty on constants is beta times
    // the perimeter
    const Eigen::VectorXd constant =
        Eigen::VectorXd::Ones(space.dof_map().num_dofs());

    REQUIRE_THAT(constant.dot(assembler.stiffness() * constant),
                 WithinRel(penalty * perimeter, 1e-12));

    // With every cell inside, each function keeps its own index, and the
    // space holds b . x, which the penalty imposes consistently
    const Eigen::VectorXd u = patch.coefficients() * b;

    REQUIRE((assembler.stiffness() * u - assembler.load()).cwiseAbs()
                .maxCoeff() < 1e-12);

    REQUIRE_THROWS_AS(assembler.assemble_load(condition, quadrature,
                                              data.head(data.size() - 1)),
                      std::invalid_argument);
}

TEST_CASE("A Neumann condition loads a trace with data along a boundary",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(
        basis, with_last_column(CellType::inside));

    const std::array<Eigen::RowVector2d, 4> corners = quadrilateral();
    const iguana::BoundaryQuadrature<double, 2> quadrature =
        boundary_quadrature(patch, corners);

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::NeumannCondition<Basis, 2> condition(trace);

    // The data h = b . x, the points lying in parameter space, where the map
    // scales each axis by its side
    const Eigen::RowVector2d b(1., -2.);
    const Eigen::VectorXd data =
        width * b(0) * quadrature.points().col(0)
        + height * b(1) * quadrature.points().col(1);

    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(condition, quadrature);
    assembler.assemble_load(condition, quadrature, data);

    // The data is known, so the stiffness stays zero
    REQUIRE(assembler.stiffness().norm() == 0.);

    // The functions sum to one, so the load sums to the integral of b . x
    // along the boundary, linear on each side: its length times the value
    // at its midpoint
    double integral = 0.;

    for (std::size_t side = 0; side < 4; ++side) {
        const Eigen::RowVector2d& from = corners[side];
        const Eigen::RowVector2d& to = corners[(side + 1) % 4];

        integral += (to - from).norm() * (from + to).dot(b) / 2.;
    }

    REQUIRE_THAT(assembler.load().sum(), WithinRel(integral, 1e-12));
}

TEST_CASE("A shifted penalty imposes a field of the space from its closest "
          "points", "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(
        basis, with_last_column(CellType::inside));

    // The surrogate domain is the element [.4, 1.2] x [0, 2.1], inside the
    // rectangle [.3, 1.3] x [-.2, 2.2], whose sides cross no knot line
    std::vector<CellType> cell_types(6, CellType::outside);
    cell_types[1] = CellType::inside;

    const iguana::SurrogateBoundary<double, 2> surrogate(
        basis.grid(), EmbeddedDomain<double, 2>(cell_types));

    const std::array<Eigen::RowVector2d, 4> corners{
        Eigen::RowVector2d(.3, -.2), Eigen::RowVector2d(1.3, -.2),
        Eigen::RowVector2d(1.3, 2.2), Eigen::RowVector2d(.3, 2.2)};
    std::vector<iguana::Boundary<double, 2>::Face> faces;

    for (std::size_t side = 0; side < 4; ++side)
        faces.push_back(segment(corners[side], corners[(side + 1) % 4]));

    const iguana::Boundary<double, 2> boundary(faces, {1, 1, 1, 1});

    const iguana::BoundaryQuadrature<double, 2> unshifted(
        surrogate, iguana::GaussLegendre<double, 1>(3));
    iguana::BoundaryQuadrature<double, 2> shifted = unshifted;
    shifted.shift(patch, boundary, 2);

    // The data of b . x at the closest points, the points lying in parameter
    // space, where the map scales each axis by its side
    const Eigen::Vector2d b(1., -2.);
    const Eigen::MatrixXd closest = shifted.points() + shifted.distances();
    const Eigen::VectorXd data = width * b(0) * closest.col(0)
                                 + height * b(1) * closest.col(1);

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PenaltyCondition<Basis, 2> condition(trace, 10.);

    // With every cell inside, each function keeps its own index, and the
    // coefficients of b . x are b . x_i at the control points
    const Eigen::VectorXd u = patch.coefficients() * b;

    // The series of b . x from each point reaches the closest point exactly,
    // so the shifted penalty imposes it consistently
    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(condition, shifted);
    assembler.assemble_load(condition, shifted, data);

    REQUIRE((assembler.stiffness() * u - assembler.load()).cwiseAbs()
                .maxCoeff() < 1e-12);

    // Without the shift, the data is imposed a distance away from where it
    // belongs
    assembler.clear();
    assembler.assemble_stiffness(condition, unshifted);
    assembler.assemble_load(condition, unshifted, data);

    REQUIRE((assembler.stiffness() * u - assembler.load()).cwiseAbs()
                .maxCoeff() > 1e-2);
}
