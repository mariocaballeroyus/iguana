/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
#include <array>
#include <cmath>
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
using iguana::CellClassification;
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

/// @brief Basis cubic along the first direction and quadratic along the
///        second, over the knot spans of uneven()
Basis cubic_by_quadratic()
{
    return Basis(TensorGrid<double, 2>(
        {KnotVector<double>(3, {0., 0., 0., 0., .2, .6, 1., 1., 1., 1.}),
         KnotVector<double>(2, {0., 0., 0., .7, 1., 1., 1.})}));
}

/// @brief Greville abscissa of a B-spline of degree p, the mean of the p
///        knots t_{i+1}, ..., t_{i+p}
double greville(const KnotVector<double>& knot_vector, int function)
{
    const std::vector<double>& t = knot_vector.values();
    const int degree = knot_vector.degree();
    double sum = 0.;

    for (int k = 1; k <= degree; ++k)
        sum += t[function + k];

    return sum / degree;
}

/// @brief Coefficients of (xi - s)_+^p in a univariate basis of degree p,
///        the products of (t_{i+k} - s)_+ over k = 1, ..., p by Marsden's
///        identity, for s a simple knot or the start of the knot vector,
///        where it gives those of (xi - s)^p
Eigen::VectorXd truncated_power(const KnotVector<double>& knot_vector,
                                double s)
{
    const std::vector<double>& t = knot_vector.values();
    const int degree = knot_vector.degree();
    const int count = static_cast<int>(t.size()) - degree - 1;
    Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(count);

    for (int function = 0; function < count; ++function)
        for (int k = 1; k <= degree; ++k)
            coefficients(function) *= std::max(t[function + k] - s, 0.);

    return coefficients;
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

/// @brief Classification of the six cells of uneven() as one type
CellClassification<double, 2> all_cells(CellType type)
{
    return CellClassification<double, 2>(std::vector<CellType>(6, type));
}

/// @brief Inside cells, except those of the last column along the first
///        direction, which take the given type
CellClassification<double, 2> with_last_column(CellType type)
{
    return CellClassification<double, 2>(
        {CellType::inside, CellType::inside, type,
         CellType::inside, CellType::inside, type});
}

/// @brief Gauss quadrature of the inside cells, exact for the products of
///        quadratic functions through an affine map
iguana::DomainQuadrature<double, 2> inside_quadrature(
    const Basis& basis, const CellClassification<double, 2>& classification)
{
    iguana::DomainQuadrature<double, 2> quadrature;
    quadrature.fill(basis.grid(), classification, CellType::inside,
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

/// @brief Quadratic basis of five by five elements of the unit square
Basis five_by_five()
{
    const KnotVector<double> fifths(
        2, {0., 0., 0., .2, .4, .6, .8, 1., 1., 1.});

    return Basis(TensorGrid<double, 2>({fifths, fifths}));
}

/// @brief Tilted quadrilateral inside the rectangle, counterclockwise, whose
///        sides cross the knot lines of five_by_five() and stay a cell away
///        from the edge of its grid, as the faces of the generalized shifted
///        boundary method need
iguana::Boundary<double, 2> inner_quadrilateral()
{
    const std::array<Eigen::RowVector2d, 4> corners{
        Eigen::RowVector2d(.5, .7), Eigen::RowVector2d(1.5, .8),
        Eigen::RowVector2d(1.4, 2.3), Eigen::RowVector2d(.6, 2.2)};
    std::vector<iguana::Boundary<double, 2>::Face> faces;

    for (std::size_t side = 0; side < 4; ++side)
        faces.push_back(segment(corners[side], corners[(side + 1) % 4]));

    return {faces, {1, 1, 1, 1}};
}

/// @brief Data of b . x at the closest points of a shifted quadrature, the
///        points lying in parameter space, where the map scales each axis by
///        its side
template<typename Quadrature>
Eigen::VectorXd linear_data(const Quadrature& quadrature,
                            const Eigen::Vector2d& b)
{
    const Eigen::MatrixXd closest =
        quadrature.points() + quadrature.distances();

    return width * b(0) * closest.col(0) + height * b(1) * closest.col(1);
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
    const CellClassification<double, 2> classification =
        with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, classification);
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
    const CellClassification<double, 2> classification =
        with_last_column(CellType::outside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, classification);
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
    const CellClassification<double, 2> classification =
        with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, classification);
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
    const CellClassification<double, 2> classification =
        with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, classification);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);
    // One value short of the points
    const Eigen::VectorXd source =
        Eigen::VectorXd::Ones(quadrature.num_points() - 1);

    REQUIRE_THROWS_AS(assembler.assemble_load(element, quadrature, source),
                      std::invalid_argument);
}

TEST_CASE("Cell weights weight the test functions of each cell",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);

    // Every cell of the space keeps its functions, and the quadrature holds
    // them type after type, in another order than the grid's
    const std::vector<CellType> cell_types{
        CellType::inside, CellType::cut, CellType::outside,
        CellType::cut, CellType::inside, CellType::outside};
    const CellClassification<double, 2> classification(cell_types);
    const iguana::FunctionSpace<Basis> space(basis,
                                             all_cells(CellType::inside));

    iguana::DomainQuadrature<double, 2> quadrature;

    for (const CellType type :
         {CellType::inside, CellType::cut, CellType::outside})
        quadrature.fill(basis.grid(), classification, type,
                        iguana::GaussLegendre<double, 2>(3));

    // Weights such as volume fractions, zero on the outside cells
    Eigen::VectorXd cell_weights(6);
    cell_weights << 1., .3, 0., .6, 1., 0.;

    const iguana::PoissonElement<Basis, 2> element;
    const Eigen::VectorXd source =
        Eigen::VectorXd::Ones(quadrature.num_points());

    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(element, quadrature, cell_weights);
    assembler.assemble_load(element, quadrature, source, cell_weights);

    // The cells span .2, .4 and .4 along x, scaled by the width, and .7 and
    // .3 along y, scaled by the height
    const std::array<double, 3> widths{.2 * width, .4 * width, .4 * width};
    const std::array<double, 2> heights{.7 * height, .3 * height};
    double weighted_area = 0.;

    for (int cell = 0; cell < 6; ++cell)
        weighted_area += cell_weights(cell) * widths[cell % 3]
                         * heights[cell / 3];

    // The functions sum to one, so the load of a unit source is the weighted
    // area, and so is the energy of x, whose gradient has unit length
    const Eigen::VectorXd x = patch.coefficients().col(0);

    REQUIRE_THAT(assembler.load().sum(), WithinRel(weighted_area, 1e-12));
    REQUIRE_THAT(x.dot(assembler.stiffness() * x),
                 WithinRel(weighted_area, 1e-12));

    // One weight short of the cells
    const Eigen::VectorXd short_weights = cell_weights.head(5);

    REQUIRE_THROWS_AS(assembler.assemble_stiffness(element, quadrature,
                                                   short_weights),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(assembler.assemble_load(element, quadrature, source,
                                              short_weights),
                      std::invalid_argument);
}

TEST_CASE("The mass of an element shares the pattern of the stiffness",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const CellClassification<double, 2> classification =
        all_cells(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::DomainQuadrature<double, 2> quadrature =
        inside_quadrature(basis, classification);
    const iguana::PoissonElement<Basis, 2> element;

    // With every cell inside, each function keeps its own index, and the
    // coefficients of x are the first coordinates of the control points
    const Eigen::VectorXd x = patch.coefficients().col(0);

    iguana::Assembler<Basis, 2> assembler(space, patch, true);
    assembler.assemble_mass(element, quadrature);

    // x^T M x is the integral of x^2 over the rectangle, into a matrix with
    // the pattern of the stiffness, faces coupled included
    REQUIRE_THAT(x.dot(assembler.mass() * x),
                 WithinRel(height * std::pow(width, 3) / 3., 1e-12));
    REQUIRE(assembler.mass().nonZeros() == assembler.stiffness().nonZeros());

    // Weighted, each cell counts its integral of x^2 by its weight, its
    // columns spanning .2, .4 and .4 along x and its rows .7 and .3 along y
    Eigen::VectorXd cell_weights(6);
    cell_weights << 1., .3, 0., .6, 1., 0.;
    const std::array<double, 4> columns{0., .2 * width, .6 * width, width};
    const std::array<double, 2> heights{.7 * height, .3 * height};
    double expected = 0.;

    for (std::size_t cell = 0; cell < 6; ++cell) {
        const double from = columns[cell % 3];
        const double to = columns[cell % 3 + 1];
        expected += cell_weights(static_cast<Eigen::Index>(cell))
                    * (to * to * to - from * from * from) / 3.
                    * heights[cell / 3];
    }

    assembler.clear();

    REQUIRE(assembler.mass().nonZeros() == assembler.stiffness().nonZeros());
    REQUIRE(assembler.mass().norm() == 0.);

    assembler.assemble_mass(element, quadrature, cell_weights);

    REQUIRE_THAT(x.dot(assembler.mass() * x), WithinRel(expected, 1e-12));

    // One weight short of the cells
    const Eigen::VectorXd short_weights = cell_weights.head(5);

    REQUIRE_THROWS_AS(assembler.assemble_mass(element, quadrature,
                                              short_weights),
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
        basis.grid(), CellClassification<double, 2>(cell_types));

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

TEST_CASE("A shifted Nitsche condition imposes a field of the space "
          "consistently", "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(
        basis, with_last_column(CellType::inside));

    // The surrogate domain is the element [.4, 1.2] x [0, 2.1], inside the
    // rectangle [.3, 1.3] x [-.2, 2.2], whose sides cross no knot line
    std::vector<CellType> cell_types(6, CellType::outside);
    cell_types[1] = CellType::inside;

    const CellClassification<double, 2> classification(cell_types);
    const iguana::SurrogateBoundary<double, 2> surrogate(basis.grid(),
                                                         classification);

    const std::array<Eigen::RowVector2d, 4> corners{
        Eigen::RowVector2d(.3, -.2), Eigen::RowVector2d(1.3, -.2),
        Eigen::RowVector2d(1.3, 2.2), Eigen::RowVector2d(.3, 2.2)};
    std::vector<iguana::Boundary<double, 2>::Face> faces;

    for (std::size_t side = 0; side < 4; ++side)
        faces.push_back(segment(corners[side], corners[(side + 1) % 4]));

    iguana::BoundaryQuadrature<double, 2> quadrature(
        surrogate, iguana::GaussLegendre<double, 1>(3));
    quadrature.shift(patch, iguana::Boundary<double, 2>(faces, {1, 1, 1, 1}),
                     2);

    // The data of b . x at the closest points, the points lying in parameter
    // space, where the map scales each axis by its side
    const Eigen::Vector2d b(1., -2.);
    const Eigen::MatrixXd closest =
        quadrature.points() + quadrature.distances();
    const Eigen::VectorXd data = width * b(0) * closest.col(0)
                                 + height * b(1) * closest.col(1);

    const iguana::PoissonElement<Basis, 2> element;
    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::DomainQuadrature<double, 2> cells =
        inside_quadrature(basis, classification);

    // With every cell inside, each function keeps its own index, and the
    // coefficients of b . x are b . x_i at the control points
    const Eigen::VectorXd u = patch.coefficients() * b;
    const Eigen::VectorXd constant =
        Eigen::VectorXd::Ones(space.dof_map().num_dofs());

    // b . x has no Laplacian, and the method is consistent, so it solves the
    // system of the surrogate domain whatever the penalty. Constants have no
    // flux, so only the penalty acts on them, gamma times the perimeter of
    // the element over its size
    for (const double penalty : {10., 1e3}) {
        INFO("penalty " << penalty);

        const iguana::NitscheCondition<Basis, 2> condition(trace, flux,
                                                           penalty);
        iguana::Assembler<Basis, 2> assembler(space, patch);
        assembler.assemble_stiffness(element, cells);
        assembler.assemble_stiffness(condition, quadrature);
        assembler.assemble_load(condition, quadrature, data);

        REQUIRE((assembler.stiffness() * u - assembler.load()).cwiseAbs()
                    .maxCoeff() < 1e-12 * penalty);
        REQUIRE_THAT(constant.dot(assembler.stiffness() * constant),
                     WithinRel(penalty * 2. * (.8 + 2.1) / std::sqrt(.8 * 2.1),
                               1e-12));
    }

    // A penalty alone leaves the flux across the surrogate boundary
    // unbalanced, however it imposes the data
    const iguana::PenaltyCondition<Basis, 2> penalty(trace, 1e3);
    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(element, cells);
    assembler.assemble_stiffness(penalty, quadrature);
    assembler.assemble_load(penalty, quadrature, data);

    REQUIRE((assembler.stiffness() * u - assembler.load()).cwiseAbs()
                .maxCoeff() > 1e-2);

    REQUIRE_THROWS_AS((iguana::NitscheCondition<Basis, 2>(trace, flux, 0.)),
                      std::invalid_argument);
}

TEST_CASE("A ghost penalty penalizes the jumps of the highest derivatives "
          "across faces", "[assembler]")
{
    const Basis basis = cubic_by_quadratic();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const KnotVector<double>& along_x = basis.axis(0).knots();
    const KnotVector<double>& along_y = basis.axis(1).knots();
    const int count = basis.axis(0).num_functions();

    // With every cell cut, every interior face is a ghost face, and with
    // every cell inside, each function keeps its own index
    const iguana::FunctionSpace<Basis> space(basis,
                                             all_cells(CellType::inside));
    const iguana::GhostFaces<double, 2> faces(basis.grid(),
                                              all_cells(CellType::cut));

    // Four points integrate the squared jumps, of degree six along the faces
    const iguana::FaceQuadrature<double, 2> quadrature(
        faces, iguana::GaussLegendre<double, 1>(4));

    const iguana::PoissonElement<Basis, 2>::U trace;
    const double penalty = .3;
    const iguana::GhostPenalty<Basis, 2> ghost(trace, penalty);

    iguana::Assembler<Basis, 2> assembler(space, patch, true);
    const Eigen::Index entries = assembler.stiffness().nonZeros();
    assembler.assemble_stiffness(ghost, quadrature);

    // The faces were coupled beforehand, so the pattern holds
    REQUIRE(assembler.stiffness().nonZeros() == entries);

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& stiffness =
        assembler.stiffness();

    // xi^3 eta^2 is one polynomial across every face, so it has no jumps
    const Eigen::VectorXd xs = truncated_power(along_x, 0.);
    const Eigen::VectorXd ys = truncated_power(along_y, 0.);
    Eigen::VectorXd u(basis.num_functions());

    for (int function = 0; function < basis.num_functions(); ++function)
        u(function) = xs(function % count) * ys(function / count);

    REQUIRE((stiffness * u).cwiseAbs().maxCoeff() < 1e-12);

    // (xi - .6)_+^3 + (eta - .7)_+^2, each basis summing to one, jumps in
    // its third derivative across xi = .6, by 3! / width^3 in physical
    // space, and in its second across eta = .7, by 2 / height^2
    const Eigen::VectorXd x_kink = truncated_power(along_x, .6);
    const Eigen::VectorXd y_kink = truncated_power(along_y, .7);
    Eigen::VectorXd v(basis.num_functions());

    for (int function = 0; function < basis.num_functions(); ++function)
        v(function) = x_kink(function % count) + y_kink(function / count);

    const double x_jump = 6. / std::pow(width, 3);
    const double y_jump = 2. / std::pow(height, 2);
    double expected = 0.;

    // gamma h^5 x_jump^2 over the faces of xi = .6, between cells of width
    // .4 on either side, then gamma h^3 y_jump^2 over those of eta = .7,
    // between cells of heights .7 and .3, h being the mean of their sizes
    for (const double span : {.7, .3}) {
        const double size = std::sqrt(width * .4 * height * span);
        expected += penalty * std::pow(size, 5) * x_jump * x_jump * height
                    * span;
    }

    for (const double span : {.2, .4, .4}) {
        const double size = (std::sqrt(width * span * height * .7)
                             + std::sqrt(width * span * height * .3)) / 2.;
        expected += penalty * std::pow(size, 3) * y_jump * y_jump * width
                    * span;
    }

    REQUIRE_THAT(v.dot(stiffness * v), WithinRel(expected, 1e-12));
}

TEST_CASE("A ghost penalty needs coupled faces and an affine B-spline patch",
          "[assembler]")
{
    const Basis basis = cubic_by_quadratic();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(basis,
                                             all_cells(CellType::inside));
    const iguana::FaceQuadrature<double, 2> quadrature(
        iguana::GhostFaces<double, 2>(basis.grid(), all_cells(CellType::cut)),
        iguana::GaussLegendre<double, 1>(4));

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::GhostPenalty<Basis, 2> ghost(trace, 1.);

    REQUIRE_THROWS_AS((iguana::GhostPenalty<Basis, 2>(trace, 0.)),
                      std::invalid_argument);

    // Faces left out of the pattern
    iguana::Assembler<Basis, 2> uncoupled(space, patch);

    REQUIRE_THROWS_AS(uncoupled.assemble_stiffness(ghost, quadrature),
                      std::invalid_argument);

    // A control point moved off the rectangle bends the map
    iguana::PointMatrix<double, 2> bent = patch.coefficients();
    bent(7, 1) += .1;
    const iguana::Patch<Basis, 2> curved(basis, bent);
    iguana::Assembler<Basis, 2> on_curved(space, curved, true);

    REQUIRE_THROWS_AS(on_curved.assemble_stiffness(ghost, quadrature),
                      std::invalid_argument);

    // A NURBS patch, affine with unit weights
    using Rational = iguana::TensorNURBS<double, 2>;
    const Rational rational(basis,
                            Eigen::VectorXd::Ones(basis.num_functions()));
    const iguana::Patch<Rational, 2> rational_patch(rational,
                                                    patch.coefficients());
    const iguana::FunctionSpace<Rational> rational_space(
        rational, all_cells(CellType::inside));
    const iguana::PoissonElement<Rational, 2>::U rational_trace;
    const iguana::GhostPenalty<Rational, 2> rational_ghost(rational_trace, 1.);
    iguana::Assembler<Rational, 2> on_rational(rational_space, rational_patch,
                                               true);

    REQUIRE_THROWS_AS(on_rational.assemble_stiffness(rational_ghost,
                                                     quadrature),
                      std::invalid_argument);
}

TEST_CASE("Face Nitsche conditions with binary weights are Nitsche "
          "conditions on the surrogate boundary", "[assembler]")
{
    const Basis basis = five_by_five();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::Boundary<double, 2> boundary = inner_quadrilateral();
    const int count = basis.grid().num_elements();

    // Weights of one on the inside cells and zero elsewhere, on a space
    // that keeps every function
    const Eigen::VectorXd fractions =
        iguana::volume_fractions(patch, boundary);
    std::vector<CellType> cell_types(static_cast<std::size_t>(count));
    Eigen::VectorXd cell_weights(count);

    for (int cell = 0; cell < count; ++cell) {
        const bool inside = fractions(cell) == 1.;
        cell_types[static_cast<std::size_t>(cell)] =
            inside ? CellType::inside : CellType::outside;
        cell_weights(cell) = inside ? 1. : 0.;
    }

    const CellClassification<double, 2> classification(cell_types);
    const iguana::FunctionSpace<Basis> space(
        basis, CellClassification<double, 2>(
                   std::vector<CellType>(cell_types.size(),
                                         CellType::inside)));

    iguana::BoundaryQuadrature<double, 2> surrogate(
        iguana::SurrogateBoundary<double, 2>(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));
    surrogate.shift(patch, boundary, 2);

    iguana::FaceQuadrature<double, 2> faces(
        iguana::JumpFaces<double, 2>(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));
    faces.shift(patch, boundary, 2);

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::NitscheCondition<Basis, 2> nitsche(trace, flux, 40.);
    const iguana::FaceNitscheCondition<Basis, 2> face(trace, flux, 40.);
    const Eigen::Vector2d b(1., -2.);

    iguana::Assembler<Basis, 2> on_surrogate(space, patch, true);
    on_surrogate.assemble_stiffness(nitsche, surrogate);
    on_surrogate.assemble_load(nitsche, surrogate, linear_data(surrogate, b));

    iguana::Assembler<Basis, 2> on_faces(space, patch, true);
    on_faces.assemble_stiffness(face, faces, cell_weights);
    on_faces.assemble_load(face, faces, linear_data(faces, b), cell_weights);

    // The jump selects the inside cell of every face of the surrogate
    // boundary, whichever side it lies on, and with it its normal
    const Eigen::MatrixXd stiffness = on_surrogate.stiffness().toDense();
    const Eigen::MatrixXd difference =
        (on_surrogate.stiffness() - on_faces.stiffness()).toDense();

    REQUIRE(difference.cwiseAbs().maxCoeff()
            < 1e-12 * stiffness.cwiseAbs().maxCoeff());
    REQUIRE((on_surrogate.load() - on_faces.load()).cwiseAbs().maxCoeff()
            < 1e-12 * on_surrogate.load().cwiseAbs().maxCoeff());
}

TEST_CASE("A linear field solves the weighted problem with Nitsche "
          "conditions on faces", "[assembler]")
{
    const Basis basis = five_by_five();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::Boundary<double, 2> boundary = inner_quadrilateral();
    const int count = basis.grid().num_elements();

    // The volume fractions weigh the cells, of a space that keeps every
    // function and a quadrature over every cell
    const Eigen::VectorXd fractions =
        iguana::volume_fractions(patch, boundary);
    const CellClassification<double, 2> classification(
        iguana::cell_types(fractions));
    const iguana::FunctionSpace<Basis> space(
        basis, CellClassification<double, 2>(std::vector<CellType>(
                   static_cast<std::size_t>(count), CellType::inside)));

    iguana::DomainQuadrature<double, 2> cells;

    for (const CellType type :
         {CellType::inside, CellType::cut, CellType::outside})
        cells.fill(basis.grid(), classification, type,
                   iguana::GaussLegendre<double, 2>(3));

    iguana::FaceQuadrature<double, 2> faces(
        iguana::JumpFaces<double, 2>(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));
    faces.shift(patch, boundary, 2);

    const iguana::FaceQuadrature<double, 2> ghost_faces(
        iguana::GhostFaces<double, 2>(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));

    const iguana::PoissonElement<Basis, 2> element;
    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::FaceNitscheCondition<Basis, 2> face(trace, flux, 40.);
    const iguana::GhostPenalty<Basis, 2> ghost(trace, .1);
    const Eigen::Vector2d b(1., -2.);

    iguana::Assembler<Basis, 2> assembler(space, patch, true);
    assembler.assemble_stiffness(element, cells, fractions);
    assembler.assemble_load(element, cells,
                            Eigen::VectorXd::Zero(cells.num_points()),
                            fractions);
    assembler.assemble_stiffness(face, faces, fractions);
    assembler.assemble_load(face, faces, linear_data(faces, b), fractions);
    assembler.assemble_stiffness(ghost, ghost_faces);

    // Cell by cell, the weighted volume terms of b . x leave the jumps of
    // the weights times its flux on the faces, which the condition cancels,
    // and its expansions reach the data exactly. With every cell inside,
    // each function keeps its own index
    const Eigen::VectorXd u = patch.coefficients() * b;

    REQUIRE((assembler.stiffness() * u - assembler.load()).cwiseAbs()
                .maxCoeff()
            < 1e-12 * assembler.load().cwiseAbs().maxCoeff());
}

TEST_CASE("Face conditions need coupled faces and one weight per cell",
          "[assembler]")
{
    const Basis basis = five_by_five();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::Boundary<double, 2> boundary = inner_quadrilateral();
    const int count = basis.grid().num_elements();

    const Eigen::VectorXd fractions =
        iguana::volume_fractions(patch, boundary);
    const std::vector<CellType> every(static_cast<std::size_t>(count),
                                      CellType::inside);
    const iguana::FunctionSpace<Basis> space(
        basis, CellClassification<double, 2>(every));

    iguana::FaceQuadrature<double, 2> faces(
        iguana::JumpFaces<double, 2>(
            basis.grid(),
            CellClassification<double, 2>(iguana::cell_types(fractions))),
        iguana::GaussLegendre<double, 1>(3));
    faces.shift(patch, boundary, 2);

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::FaceNitscheCondition<Basis, 2> face(trace, flux, 40.);
    const Eigen::VectorXd data = Eigen::VectorXd::Zero(faces.num_points());

    REQUIRE_THROWS_AS(
        (iguana::FaceNitscheCondition<Basis, 2>(trace, flux, 0.)),
        std::invalid_argument);

    // Faces left out of the pattern
    iguana::Assembler<Basis, 2> uncoupled(space, patch);

    REQUIRE_THROWS_AS(uncoupled.assemble_stiffness(face, faces, fractions),
                      std::invalid_argument);

    // One weight short of the cells, and one value short of the points
    iguana::Assembler<Basis, 2> assembler(space, patch, true);
    const Eigen::VectorXd short_weights = fractions.head(count - 1);

    REQUIRE_THROWS_AS(assembler.assemble_stiffness(face, faces,
                                                   short_weights),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(assembler.assemble_load(face, faces, data,
                                              short_weights),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(assembler.assemble_load(face, faces, data.head(1),
                                              fractions),
                      std::invalid_argument);

    // A NURBS patch has no Taylor series for a shifted quadrature
    using Rational = iguana::TensorNURBS<double, 2>;
    const Rational rational(basis,
                            Eigen::VectorXd::Ones(basis.num_functions()));
    const iguana::Patch<Rational, 2> rational_patch(rational,
                                                    patch.coefficients());
    const iguana::FunctionSpace<Rational> rational_space(
        rational, CellClassification<double, 2>(every));
    const iguana::PoissonElement<Rational, 2>::U rational_trace;
    const iguana::PoissonElement<Rational, 2>::Q rational_flux;
    const iguana::FaceNitscheCondition<Rational, 2> rational_face(
        rational_trace, rational_flux, 40.);
    iguana::Assembler<Rational, 2> on_rational(rational_space, rational_patch,
                                               true);

    REQUIRE_THROWS_AS(on_rational.assemble_stiffness(rational_face, faces,
                                                     fractions),
                      std::invalid_argument);
}
