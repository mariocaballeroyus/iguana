/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <petscmat.h>

namespace
{

using Catch::Matchers::WithinRel;
using iguana::CellType;
using iguana::CellClassification;
using iguana::KnotVector;
using iguana::TensorGrid;
using Basis = iguana::TensorBSpline<double, 2>;

/// @brief Throws when a PETSc call fails, which fails the test
void check(PetscErrorCode code)
{
    if (code != PETSC_SUCCESS)
        throw std::runtime_error("a PETSc call failed");
}

/// @brief Entries of a PETSc matrix, the explicit zeros of its pattern
///        included
Eigen::SparseMatrix<double, Eigen::RowMajor> to_eigen(Mat matrix)
{
    PetscInt rows = 0;
    PetscInt cols = 0;
    check(MatGetSize(matrix, &rows, &cols));

    std::vector<Eigen::Triplet<double>> entries;

    for (PetscInt row = 0; row < rows; ++row) {
        PetscInt count = 0;
        const PetscInt* columns = nullptr;
        const PetscScalar* values = nullptr;
        check(MatGetRow(matrix, row, &count, &columns, &values));

        for (PetscInt entry = 0; entry < count; ++entry)
            entries.emplace_back(row, columns[entry], values[entry]);

        check(MatRestoreRow(matrix, row, &count, &columns, &values));
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor> result(rows, cols);
    result.setFromTriplets(entries.begin(), entries.end());

    return result;
}

/// @brief Entries of a PETSc vector
Eigen::VectorXd to_eigen(Vec vector)
{
    PetscInt size = 0;
    check(VecGetSize(vector, &size));

    const PetscScalar* values = nullptr;
    check(VecGetArrayRead(vector, &values));
    const Eigen::VectorXd result = Eigen::Map<const Eigen::VectorXd>(values,
                                                                     size);
    check(VecRestoreArrayRead(vector, &values));

    return result;
}

/// @brief Stiffness of an assembler, read back from PETSc
template<typename A>
Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness_of(const A& assembler)
{
    return to_eigen(assembler.stiffness());
}

/// @brief Mass of an assembler, read back from PETSc
template<typename A>
Eigen::SparseMatrix<double, Eigen::RowMajor> mass_of(const A& assembler)
{
    return to_eigen(assembler.mass());
}

/// @brief Load of an assembler, read back from PETSc
template<typename A>
Eigen::VectorXd load_of(const A& assembler)
{
    return to_eigen(assembler.load());
}

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

/// @brief Basis quadratic along the first direction and cubic along the
///        second, over the knot spans of uneven()
Basis quadratic_by_cubic()
{
    return Basis(TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., .2, .6, 1., 1., 1.}),
         KnotVector<double>(3, {0., 0., 0., 0., .7, 1., 1., 1., 1.})}));
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
iguana::QuadratureDomain<double, 2> inside_quadrature(
    const Basis& basis, const CellClassification<double, 2>& classification)
{
    iguana::QuadratureDomain<double, 2> quadrature;
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

/// @brief Data of b . x at the closest points of some points, which a
///        projection gives in parameter space, where the map scales each axis
///        by its side
Eigen::VectorXd linear_data(
    const Eigen::MatrixXd& points,
    const iguana::BoundaryProjection<double>& projection,
    const Eigen::Vector2d& b)
{
    const Eigen::MatrixXd closest = points + projection.distances;

    return width * b(0) * closest.col(0) + height * b(1) * closest.col(1);
}

/// @brief Quadrature of the boundary of a quadrilateral in a patch, with
///        five points per piece, as biquadratic functions are of degree four
///        along a line and their products integrate exactly
iguana::QuadratureBoundary<double, 2> boundary_quadrature(
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

/// @brief Greville abscissae of every function of a univariate basis
Eigen::VectorXd grevilles(const KnotVector<double>& knot_vector)
{
    const int count = static_cast<int>(knot_vector.values().size())
                      - knot_vector.degree() - 1;
    Eigen::VectorXd result(count);

    for (int function = 0; function < count; ++function)
        result(function) = greville(knot_vector, function);

    return result;
}

/// @brief Coefficients of xi^2 in a univariate basis of degree p, at least
///        two, its blossom at the knots t_{i+1}, ..., t_{i+p}: the mean of
///        the products of their pairs
Eigen::VectorXd squares(const KnotVector<double>& knot_vector)
{
    const std::vector<double>& t = knot_vector.values();
    const int degree = knot_vector.degree();
    const int count = static_cast<int>(t.size()) - degree - 1;
    const double pairs = degree * (degree - 1) / 2.;
    Eigen::VectorXd result = Eigen::VectorXd::Zero(count);

    for (int function = 0; function < count; ++function) {
        for (int k = 1; k <= degree; ++k) {
            for (int l = k + 1; l <= degree; ++l)
                result(function) += t[function + k] * t[function + l];
        }

        result(function) /= pairs;
    }

    return result;
}

/// @brief Coefficients of a product of a polynomial along x and one along
///        y in a basis on the elements of rectangle(), given by their
///        coefficients along each direction, the first running fastest
Eigen::VectorXd product(const Eigen::VectorXd& along_x,
                        const Eigen::VectorXd& along_y)
{
    return (along_x * along_y.transpose()).reshaped();
}

/// @brief Coefficients of the flow u = (x^2, −2 x y), which has no
///        divergence, with the pressure p = x + y, in a space of block size
///        three on a rectangle() with every cell inside: those of x^2 and
///        x y from the blossoms of each direction, those of x + y at the
///        control points, component c of function k at 3 k + c
Eigen::VectorXd quadratic_flow(const iguana::Patch<Basis, 2>& patch)
{
    const Basis& basis = patch.basis();
    const KnotVector<double>& along_x = basis.grid().knots(0);
    const KnotVector<double>& along_y = basis.grid().knots(1);
    const Eigen::VectorXd ones_y =
        Eigen::VectorXd::Ones(basis.axis(1).num_functions());

    Eigen::MatrixXd components(3, basis.num_functions());
    components.row(0) =
        product(width * width * truncated_power(along_x, 0.), ones_y)
            .transpose();
    components.row(1) = product(-2. * width * grevilles(along_x),
                                height * grevilles(along_y))
                            .transpose();
    components.row(2) =
        (patch.coefficients() * Eigen::Vector2d(1., 1.)).transpose();

    return components.reshaped();
}

/// @brief Trace of one component of a field of two, the values of the
///        functions in the rows of that component and zeros in the others
class Component final : public iguana::Trace<Basis, 2>
{
public:
    explicit Component(Eigen::Index component) : component_(component)
    {
    }

    iguana::ValueFlags flags() const noexcept override
    { return {}; }

    std::size_t field(std::size_t) const noexcept override
    { return 0; }

    void local_trace(const iguana::ElementValues<Basis, 2>& values,
                     Eigen::MatrixXd& trace) const override
    {
        const Eigen::Index rows = values.values().rows();
        trace.setZero(2 * rows, values.values().cols());
        trace.middleRows(component_ * rows, rows) = values.values();
    }

private:
    Eigen::Index component_;
};

/// @brief Coefficients of the field A x in a space of block size two on a
///        rectangle with every cell inside, its values at the control points
///        through the affine map, component c of function k at 2 k + c, the
///        column-major order of a matrix of a column per function
Eigen::VectorXd linear_vector_field(const iguana::Patch<Basis, 2>& patch,
                                    const Eigen::Matrix2d& a)
{
    const Eigen::MatrixXd components =
        (patch.coefficients() * a.transpose()).transpose();

    return components.reshaped();
}

/// @brief Trace of one field of a mixed space whose fields have one
///        component each, the values of its functions in its rows, after
///        those of the fields before it, and zeros in the others
class FieldTrace final : public iguana::Trace<Basis, 2>
{
public:
    explicit FieldTrace(std::size_t field) : field_(field)
    {
    }

    iguana::ValueFlags flags() const noexcept override
    { return {}; }

    std::size_t field(std::size_t) const noexcept override
    { return field_; }

    void local_trace(const iguana::ElementValues<Basis, 2>& values,
                     Eigen::MatrixXd& trace) const override
    {
        Eigen::Index offset = 0;
        Eigen::Index rows = 0;

        for (std::size_t other = 0; other < values.num_fields(); ++other) {
            if (other < field_)
                offset += values.values(other).rows();

            rows += values.values(other).rows();
        }

        const Eigen::MatrixXd& own = values.values(field_);
        trace.setZero(rows, own.cols());
        trace.middleRows(offset, own.rows()) = own;
    }

private:
    std::size_t field_;
};

/// @brief Matrix that places the degrees of freedom of a field at its range
///        of a mixed space, with a one at (offset + i, i)
Eigen::MatrixXd placement(int num_dofs, int offset, int count)
{
    Eigen::MatrixXd place = Eigen::MatrixXd::Zero(num_dofs, count);
    place.middleRows(offset, count).setIdentity();

    return place;
}

/// @brief Whether a matrix has the entries of another, up to rounding
///        relative to the largest
bool near(const Eigen::MatrixXd& computed, const Eigen::MatrixXd& expected)
{
    return (computed - expected).cwiseAbs().maxCoeff()
           <= 1e-12 * expected.cwiseAbs().maxCoeff();
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
    const iguana::QuadratureDomain<double, 2> quadrature =
        inside_quadrature(basis, classification);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);

    // The source f = y, with y = height times the second parameter
    const Eigen::VectorXd source = height * quadrature.points().col(1);

    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, source);

    const Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness =
        stiffness_of(assembler);

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
    REQUIRE_THAT(load_of(assembler).sum(),
                 WithinRel(width * height * height / 2., 1e-12));
}

TEST_CASE("A space of another basis on the elements of the patch assembles "
          "the Poisson system", "[assembler]")
{
    // A space cubic along the first direction on the quadratic rectangle,
    // whose elements it shares
    const iguana::Patch<Basis, 2> patch = rectangle(uneven());
    const Basis basis = cubic_by_quadratic();
    const CellClassification<double, 2> classification =
        all_cells(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::QuadratureDomain<double, 2> quadrature =
        inside_quadrature(basis, classification);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);

    // The source f = y, with y = height times the second parameter
    const Eigen::VectorXd source = height * quadrature.points().col(1);

    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, source);

    const Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness =
        stiffness_of(assembler);

    // Every spline space holds the linear fields, whose coefficients are
    // b . x at the Greville points of its own basis
    const Eigen::Vector2d b(1., -2.);
    const int count = basis.axis(0).num_functions();
    Eigen::VectorXd u(basis.num_functions());

    for (int function = 0; function < basis.num_functions(); ++function)
        u(function) =
            b(0) * width * greville(basis.grid().knots(0), function % count)
            + b(1) * height
                  * greville(basis.grid().knots(1), function / count);

    REQUIRE_THAT(u.dot(stiffness * u),
                 WithinRel(b.squaredNorm() * width * height, 1e-12));

    const Eigen::VectorXd constant =
        Eigen::VectorXd::Ones(space.dof_map().num_dofs());

    REQUIRE((stiffness * constant).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE_THAT(load_of(assembler).sum(),
                 WithinRel(width * height * height / 2., 1e-12));
}

TEST_CASE("An assembler needs a space on the elements of the patch",
          "[assembler]")
{
    const iguana::Patch<Basis, 2> patch = rectangle(uneven());
    const iguana::FunctionSpace<Basis> space(
        five_by_five(),
        CellClassification<double, 2>(
            std::vector<CellType>(25, CellType::inside)));

    REQUIRE_THROWS_AS((iguana::Assembler<Basis, 2>(space, patch)),
                      std::invalid_argument);
}

TEST_CASE("The Poisson system leaves out the functions of outside cells",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const CellClassification<double, 2> classification =
        with_last_column(CellType::outside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::QuadratureDomain<double, 2> quadrature =
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

    REQUIRE((stiffness_of(assembler) * constant).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE_THAT(load_of(assembler).sum(),
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
    const iguana::QuadratureDomain<double, 2> quadrature =
        inside_quadrature(basis, classification);
    const iguana::PoissonElement<Basis, 2> element;

    iguana::Assembler<Basis, 2> assembler(space, patch);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(quadrature.num_points());

    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, ones);

    const Eigen::MatrixXd stiffness(stiffness_of(assembler));
    const Eigen::VectorXd load = load_of(assembler);

    // Assembling again after clearing gives the same system, not twice it
    assembler.clear();
    assembler.assemble_stiffness(element, quadrature);
    assembler.assemble_load(element, quadrature, ones);

    const Eigen::MatrixXd again(stiffness_of(assembler));

    REQUIRE((again - stiffness).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE((load_of(assembler) - load).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("A form adds its integrals in one assembly, with its data as it "
          "is then", "[assembler][form]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const CellClassification<double, 2> classification =
        with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::QuadratureDomain<double, 2> quadrature =
        inside_quadrature(basis, classification);
    const iguana::PoissonElement<Basis, 2> element;

    Eigen::VectorXd source = Eigen::VectorXd::Ones(quadrature.num_points());
    const Eigen::VectorXd weights = Eigen::VectorXd::Constant(6, .5);

    iguana::Form<Basis, 2> form;
    form.add_domain_integral(element, quadrature, source, weights);
    form.add_mass_integral(element, quadrature, weights);

    // The form reads the source where it lies, so values set after it was
    // built reach the assembly
    source.setLinSpaced(0., 1.);

    iguana::Assembler<Basis, 2> by_form(space, patch);
    form.assemble(by_form);

    iguana::Assembler<Basis, 2> by_terms(space, patch);
    by_terms.assemble_stiffness(element, quadrature, weights);
    by_terms.assemble_load(element, quadrature, source, weights);
    by_terms.assemble_mass(element, quadrature, weights);

    REQUIRE((stiffness_of(by_form) - stiffness_of(by_terms)).norm() == 0.);
    REQUIRE((mass_of(by_form) - mass_of(by_terms)).norm() == 0.);
    REQUIRE((load_of(by_form) - load_of(by_terms)).norm() == 0.);
}

TEST_CASE("An assembler needs one source value per point", "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const CellClassification<double, 2> classification =
        with_last_column(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::QuadratureDomain<double, 2> quadrature =
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

    iguana::QuadratureDomain<double, 2> quadrature;

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

    REQUIRE_THAT(load_of(assembler).sum(), WithinRel(weighted_area, 1e-12));
    REQUIRE_THAT(x.dot(stiffness_of(assembler) * x),
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

TEST_CASE("The pattern pairs the degrees of freedom of each element, and of "
          "the two cells of each face when coupled", "[assembler]")
{
    const iguana::Patch<Basis, 2> patch = rectangle(uneven());

    // A field of two components, cubic by quadratic, then a quadratic one,
    // with the last column of cells outside, where they have no degrees of
    // freedom
    const CellClassification<double, 2> classification =
        with_last_column(CellType::outside);
    const iguana::FunctionSpace<Basis> vector(cubic_by_quadratic(),
                                              classification, 2);
    const iguana::FunctionSpace<Basis> scalar(uneven(), classification);
    const iguana::MixedFunctionSpace<Basis> space({vector, scalar});
    const int num_dofs = space.num_dofs();

    for (const bool couple_faces : {false, true}) {
        // Every pair on a cell and, with the faces coupled, on the cell and
        // the next one along each direction, cell = column + 3 row
        Eigen::MatrixX<bool> expected =
            Eigen::MatrixX<bool>::Constant(num_dofs, num_dofs, false);
        std::vector<int> dofs;
        std::vector<int> next;

        for (int cell = 0; cell < 6; ++cell) {
            space.dofs_on_element(cell, dofs);

            for (const int row : dofs) {
                for (const int col : dofs)
                    expected(row, col) = true;
            }

            std::vector<int> neighbours;

            if (couple_faces && cell % 3 < 2)
                neighbours.push_back(cell + 1);

            if (couple_faces && cell < 3)
                neighbours.push_back(cell + 3);

            for (const int neighbour : neighbours) {
                space.dofs_on_element(neighbour, next);

                for (const int row : dofs) {
                    for (const int col : next) {
                        expected(row, col) = true;
                        expected(col, row) = true;
                    }
                }
            }
        }

        // The explicit zeros of the stiffness before any assembly
        const iguana::Assembler<Basis, 2> assembler(space, patch,
                                                    couple_faces);
        const Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness =
            stiffness_of(assembler);
        Eigen::MatrixX<bool> pattern =
            Eigen::MatrixX<bool>::Constant(num_dofs, num_dofs, false);

        for (int row = 0; row < num_dofs; ++row) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator
                     entry(stiffness, row);
                 entry; ++entry)
                pattern(entry.row(), entry.col()) = true;
        }

        REQUIRE(pattern == expected);
    }
}

TEST_CASE("The mass of an element shares the pattern of the stiffness",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const CellClassification<double, 2> classification =
        all_cells(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification);
    const iguana::QuadratureDomain<double, 2> quadrature =
        inside_quadrature(basis, classification);
    const iguana::PoissonElement<Basis, 2> element;

    // With every cell inside, each function keeps its own index, and the
    // coefficients of x are the first coordinates of the control points
    const Eigen::VectorXd x = patch.coefficients().col(0);

    iguana::Assembler<Basis, 2> assembler(space, patch, true);
    assembler.assemble_mass(element, quadrature);

    // x^T M x is the integral of x^2 over the rectangle, into a matrix with
    // the pattern of the stiffness, faces coupled included
    REQUIRE_THAT(x.dot(mass_of(assembler) * x),
                 WithinRel(height * std::pow(width, 3) / 3., 1e-12));
    REQUIRE(mass_of(assembler).nonZeros()
            == stiffness_of(assembler).nonZeros());

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

    REQUIRE(mass_of(assembler).nonZeros()
            == stiffness_of(assembler).nonZeros());
    REQUIRE(mass_of(assembler).norm() == 0.);

    assembler.assemble_mass(element, quadrature, cell_weights);

    REQUIRE_THAT(x.dot(mass_of(assembler) * x), WithinRel(expected, 1e-12));

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
    const iguana::QuadratureBoundary<double, 2> quadrature =
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

    REQUIRE_THAT(constant.dot(stiffness_of(assembler) * constant),
                 WithinRel(penalty * perimeter, 1e-12));

    // With every cell inside, each function keeps its own index, and the
    // space holds b . x, which the penalty imposes consistently
    const Eigen::VectorXd u = patch.coefficients() * b;

    REQUIRE((stiffness_of(assembler) * u - load_of(assembler)).cwiseAbs()
                .maxCoeff() < 1e-12);

    REQUIRE_THROWS_AS(assembler.assemble_load(condition, quadrature,
                                              data.head(data.size() - 1)),
                      std::invalid_argument);
}

TEST_CASE("A penalty on one component of a field of two imposes it and "
          "leaves the other free", "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(basis,
                                             all_cells(CellType::inside), 2);

    const std::array<Eigen::RowVector2d, 4> corners = quadrilateral();
    const iguana::QuadratureBoundary<double, 2> quadrature =
        boundary_quadrature(patch, corners);
    double perimeter = 0.;

    for (std::size_t side = 0; side < 4; ++side)
        perimeter += (corners[(side + 1) % 4] - corners[side]).norm();

    const Component second(1);
    const double penalty = 10.;
    const iguana::PenaltyCondition<Basis, 2> condition(second, penalty);

    // The data of b . x for the second component, the points lying in
    // parameter space, where the map scales each axis by its side
    const Eigen::Vector2d b(1., -2.);
    const Eigen::VectorXd data =
        width * b(0) * quadrature.points().col(0)
        + height * b(1) * quadrature.points().col(1);

    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(condition, quadrature);
    assembler.assemble_load(condition, quadrature, data);

    const Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness =
        stiffness_of(assembler);

    // The functions sum to one, so a constant second component costs beta
    // times the perimeter, while the first is not penalized at all
    const int num_functions = basis.num_functions();
    Eigen::VectorXd first_constant = Eigen::VectorXd::Zero(2 * num_functions);
    Eigen::VectorXd second_constant = first_constant;
    first_constant(Eigen::seqN(0, num_functions, 2)).setOnes();
    second_constant(Eigen::seqN(1, num_functions, 2)).setOnes();

    REQUIRE((stiffness * first_constant).norm() == 0.);
    REQUIRE_THAT(second_constant.dot(stiffness * second_constant),
                 WithinRel(penalty * perimeter, 1e-12));

    // Any first component with b . x as the second satisfies the system,
    // which imposes it consistently
    Eigen::Matrix2d a;
    a << 3., .5, b(0), b(1);
    const Eigen::VectorXd u = linear_vector_field(patch, a);

    REQUIRE((stiffness * u - load_of(assembler)).cwiseAbs().maxCoeff()
            < 1e-12);

    // The system tells PETSc which degrees of freedom are the components of
    // one function
    PetscInt block_size = 0;
    PetscInt load_size = 0;
    check(MatGetBlockSize(assembler.stiffness(), &block_size));
    check(VecGetBlockSize(assembler.load(), &load_size));

    REQUIRE(block_size == 2);
    REQUIRE(load_size == 2);
}

TEST_CASE("Local matrices must match the block size of the space",
          "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const CellClassification<double, 2> classification =
        all_cells(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification, 2);
    const iguana::QuadratureDomain<double, 2> domain =
        inside_quadrature(basis, classification);
    const iguana::QuadratureBoundary<double, 2> boundary =
        boundary_quadrature(patch, quadrilateral());
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(domain.num_points());

    // The Poisson element and its trace have one component, so their local
    // matrices hold half the degrees of freedom of each cell, and PETSc
    // would read past their end
    const iguana::PoissonElement<Basis, 2> element;
    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PenaltyCondition<Basis, 2> condition(trace, 10.);

    iguana::Assembler<Basis, 2> assembler(space, patch);

    REQUIRE_THROWS_AS(assembler.assemble_stiffness(element, domain),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(assembler.assemble_load(element, domain, ones),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(assembler.assemble_stiffness(condition, boundary),
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
    const iguana::QuadratureBoundary<double, 2> quadrature =
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
    REQUIRE(stiffness_of(assembler).norm() == 0.);

    // The functions sum to one, so the load sums to the integral of b . x
    // along the boundary, linear on each side: its length times the value
    // at its midpoint
    double integral = 0.;

    for (std::size_t side = 0; side < 4; ++side) {
        const Eigen::RowVector2d& from = corners[side];
        const Eigen::RowVector2d& to = corners[(side + 1) % 4];

        integral += (to - from).norm() * (from + to).dot(b) / 2.;
    }

    REQUIRE_THAT(load_of(assembler).sum(), WithinRel(integral, 1e-12));
}

TEST_CASE("A Neumann condition on one component of a field of two loads "
          "it alone", "[assembler]")
{
    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(basis,
                                             all_cells(CellType::inside), 2);

    const std::array<Eigen::RowVector2d, 4> corners = quadrilateral();
    const iguana::QuadratureBoundary<double, 2> quadrature =
        boundary_quadrature(patch, corners);

    const Component second(1);
    const iguana::NeumannCondition<Basis, 2> condition(second);
    const Eigen::VectorXd data = Eigen::VectorXd::Ones(quadrature.num_points());

    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(condition, quadrature);
    assembler.assemble_load(condition, quadrature, data);

    // No stiffness over the two components, and the unit data loads the
    // second alone with the perimeter, as the functions sum to one
    double perimeter = 0.;

    for (std::size_t side = 0; side < 4; ++side)
        perimeter += (corners[(side + 1) % 4] - corners[side]).norm();

    const Eigen::VectorXd load = load_of(assembler);
    const int num_functions = basis.num_functions();

    REQUIRE(stiffness_of(assembler).norm() == 0.);
    REQUIRE(load(Eigen::seqN(0, num_functions, 2)).norm() == 0.);
    REQUIRE_THAT(load(Eigen::seqN(1, num_functions, 2)).sum(),
                 WithinRel(perimeter, 1e-12));
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

    const iguana::QuadratureBoundary<double, 2> quadrature(
        surrogate, iguana::GaussLegendre<double, 1>(3));
    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(patch, boundary, quadrature.points());

    // The data of b . x at the closest points, the points lying in parameter
    // space, where the map scales each axis by its side
    const Eigen::Vector2d b(1., -2.);
    const Eigen::MatrixXd closest = quadrature.points() + projection.distances;
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
    assembler.assemble_stiffness(condition, quadrature, projection, 2);
    assembler.assemble_load(condition, quadrature, projection, 2, data);

    REQUIRE((stiffness_of(assembler) * u - load_of(assembler)).cwiseAbs()
                .maxCoeff() < 1e-12);

    // Without the shift, the data is imposed a distance away from where it
    // belongs
    assembler.clear();
    assembler.assemble_stiffness(condition, quadrature);
    assembler.assemble_load(condition, quadrature, data);

    REQUIRE((stiffness_of(assembler) * u - load_of(assembler)).cwiseAbs()
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

    const iguana::QuadratureBoundary<double, 2> quadrature(
        surrogate, iguana::GaussLegendre<double, 1>(3));
    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(
            patch, iguana::Boundary<double, 2>(faces, {1, 1, 1, 1}),
            quadrature.points());

    // The data of b . x at the closest points, the points lying in parameter
    // space, where the map scales each axis by its side
    const Eigen::Vector2d b(1., -2.);
    const Eigen::MatrixXd closest = quadrature.points() + projection.distances;
    const Eigen::VectorXd data = width * b(0) * closest.col(0)
                                 + height * b(1) * closest.col(1);

    const iguana::PoissonElement<Basis, 2> element;
    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::QuadratureDomain<double, 2> cells =
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
        assembler.assemble_stiffness(condition, quadrature, projection, 2);
        assembler.assemble_load(condition, quadrature, projection, 2, data);

        REQUIRE((stiffness_of(assembler) * u - load_of(assembler)).cwiseAbs()
                    .maxCoeff() < 1e-12 * penalty);
        REQUIRE_THAT(constant.dot(stiffness_of(assembler) * constant),
                     WithinRel(penalty * 2. * (.8 + 2.1) / std::sqrt(.8 * 2.1),
                               1e-12));
    }

    // A penalty alone leaves the flux across the surrogate boundary
    // unbalanced, however it imposes the data
    const iguana::PenaltyCondition<Basis, 2> penalty(trace, 1e3);
    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(element, cells);
    assembler.assemble_stiffness(penalty, quadrature, projection, 2);
    assembler.assemble_load(penalty, quadrature, projection, 2, data);

    REQUIRE((stiffness_of(assembler) * u - load_of(assembler)).cwiseAbs()
                .maxCoeff() > 1e-2);

    REQUIRE_THROWS_AS((iguana::NitscheCondition<Basis, 2>(trace, flux, 0.)),
                      std::invalid_argument);
}

TEST_CASE("A Stokes system with Nitsche conditions holds a divergence-free "
          "quadratic flow exactly, up to a constant pressure", "[assembler]")
{
    using Element = iguana::StokesStabilizedElement<Basis, 2>;

    const Basis basis = uneven();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const CellClassification<double, 2> classification =
        all_cells(CellType::inside);
    const iguana::FunctionSpace<Basis> space(basis, classification, 3);
    const iguana::QuadratureDomain<double, 2> cells =
        inside_quadrature(basis, classification);

    // The edges of the rectangle, the surrogate boundary of its cells
    const iguana::QuadratureBoundary<double, 2> edges(
        iguana::SurrogateBoundary<double, 2>(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));

    const double viscosity = 1.5;
    const Element element(viscosity, 3.);
    const std::array<Element::U, 2> traces{Element::U(0), Element::U(1)};
    const std::array<Element::T, 2> fluxes{Element::T(viscosity, 0),
                                           Element::T(viscosity, 1)};

    // u = (x^2, −2 x y) has no divergence, and with p = x + y the body force
    // f = −div(2μ ε(u)) + ∇p = (1 − 2μ, 1)
    Eigen::MatrixXd source(cells.num_points(), 2);
    source.col(0).setConstant(1. - 2. * viscosity);
    source.col(1).setConstant(1.);

    // The values of u along the edges, the points lying in parameter space,
    // where the map scales each axis by its side
    const Eigen::VectorXd x = width * edges.points().col(0);
    const Eigen::VectorXd y = height * edges.points().col(1);
    const std::array<Eigen::VectorXd, 2> data{x.cwiseProduct(x),
                                              -2. * x.cwiseProduct(y)};

    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(element, cells);
    assembler.assemble_load(element, cells, source);

    for (std::size_t component = 0; component < 2; ++component) {
        const iguana::NitscheCondition<Basis, 2> condition(
            traces[component], fluxes[component], 40. * viscosity);

        assembler.assemble_stiffness(condition, edges);
        assembler.assemble_load(condition, edges, data[component]);
    }

    const Eigen::VectorXd exact = quadratic_flow(patch);
    const Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness =
        stiffness_of(assembler);

    // The method and Nitsche's conditions are consistent, the residual of
    // the momentum equation vanishing
    REQUIRE((stiffness * exact - load_of(assembler)).cwiseAbs().maxCoeff()
            < 1e-11);

    // With the velocity imposed all around, a constant pressure balances
    // its own flux through the boundary, so it solves the system with zero
    Eigen::VectorXd pressure = Eigen::VectorXd::Zero(exact.size());
    pressure(Eigen::seqN(2, basis.num_functions(), 3)).setOnes();

    REQUIRE((stiffness * pressure).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("A Stokes system on divergence-conforming spaces holds a "
          "divergence-free quadratic flow exactly, up to a constant "
          "pressure", "[assembler]")
{
    using Element = iguana::StokesElement<Basis, 2>;

    // Each component of the velocity one degree higher along its own
    // direction than the pressure, on the elements of the patch
    const iguana::Patch<Basis, 2> patch = rectangle(uneven());
    const CellClassification<double, 2> classification =
        all_cells(CellType::inside);
    const iguana::FunctionSpace<Basis> x_velocity(cubic_by_quadratic(),
                                                  classification);
    const iguana::FunctionSpace<Basis> y_velocity(quadratic_by_cubic(),
                                                  classification);
    const iguana::FunctionSpace<Basis> pressure(uneven(), classification);
    const iguana::MixedFunctionSpace<Basis> space(
        {x_velocity, y_velocity, pressure});

    // Four points integrate the products of cubic functions exactly
    iguana::QuadratureDomain<double, 2> cells;
    cells.fill(patch.basis().grid(), classification, CellType::inside,
               iguana::GaussLegendre<double, 2>(4));

    // The edges of the rectangle, the surrogate boundary of its cells
    const iguana::QuadratureBoundary<double, 2> edges(
        iguana::SurrogateBoundary<double, 2>(patch.basis().grid(),
                                             classification),
        iguana::GaussLegendre<double, 1>(4));

    const double viscosity = 1.5;
    const Element element(viscosity);
    const std::array<Element::U, 2> traces{Element::U(0), Element::U(1)};
    const std::array<Element::T, 2> fluxes{Element::T(viscosity, 0),
                                           Element::T(viscosity, 1)};

    // u = (x^2, −2 x y) has no divergence, and with p = x + y the body force
    // f = −div(2μ ε(u)) + ∇p = (1 − 2μ, 1)
    Eigen::MatrixXd source(cells.num_points(), 2);
    source.col(0).setConstant(1. - 2. * viscosity);
    source.col(1).setConstant(1.);

    // The values of u along the edges, the points lying in parameter space,
    // where the map scales each axis by its side
    const Eigen::VectorXd x = width * edges.points().col(0);
    const Eigen::VectorXd y = height * edges.points().col(1);
    const std::array<Eigen::VectorXd, 2> data{x.cwiseProduct(x),
                                              -2. * x.cwiseProduct(y)};

    iguana::Assembler<Basis, 2> assembler(space, patch);
    assembler.assemble_stiffness(element, cells);
    assembler.assemble_load(element, cells, source);

    for (std::size_t component = 0; component < 2; ++component) {
        const iguana::NitscheCondition<Basis, 2> condition(
            traces[component], fluxes[component], 40. * viscosity);

        assembler.assemble_stiffness(condition, edges);
        assembler.assemble_load(condition, edges, data[component]);
    }

    // The coefficients of x^2 and x y from the blossoms of each direction,
    // and those of x + y at the control points of the patch, whose basis
    // the pressure takes, one field after another
    const Basis& x_basis = x_velocity.basis();
    const Basis& y_basis = y_velocity.basis();
    Eigen::VectorXd exact(space.num_dofs());
    exact << product(width * width * squares(x_basis.grid().knots(0)),
                     Eigen::VectorXd::Ones(x_basis.axis(1).num_functions())),
        product(-2. * width * grevilles(y_basis.grid().knots(0)),
                height * grevilles(y_basis.grid().knots(1))),
        patch.coefficients() * Eigen::Vector2d(1., 1.);

    const Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness =
        stiffness_of(assembler);

    // The method and Nitsche's conditions are consistent, the residual of
    // the momentum equation vanishing
    REQUIRE((stiffness * exact - load_of(assembler)).cwiseAbs().maxCoeff()
            < 1e-11);

    // With the velocity imposed all around, a constant pressure balances
    // its own flux through the boundary, so it solves the system with zero
    Eigen::VectorXd constant = Eigen::VectorXd::Zero(space.num_dofs());
    constant.tail(pressure.num_dofs()).setOnes();

    REQUIRE((stiffness * constant).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("A ghost penalty penalizes the jumps of the highest derivatives of "
          "the space across faces", "[assembler]")
{
    const Basis basis = cubic_by_quadratic();
    const KnotVector<double>& along_x = basis.axis(0).knots();
    const KnotVector<double>& along_y = basis.axis(1).knots();
    const int count = basis.axis(0).num_functions();

    // With every cell cut, every interior face is a ghost face, and with
    // every cell inside, each function keeps its own index
    const iguana::FunctionSpace<Basis> space(basis,
                                             all_cells(CellType::inside));
    const std::vector<int> faces =
        iguana::ghost_faces(basis.grid(), all_cells(CellType::cut));

    // Four points integrate the squared jumps, of degree six along the faces
    const iguana::QuadratureSkeleton<double, 2> quadrature(
        basis.grid(), faces, iguana::GaussLegendre<double, 1>(4));

    const iguana::PoissonElement<Basis, 2>::U trace;
    const double penalty = .3;
    const iguana::SkeletonGhostPenalty<Basis, 2> ghost(trace, penalty);

    // xi^3 eta^2 is one polynomial across every face, so it has no jumps
    const Eigen::VectorXd xs = truncated_power(along_x, 0.);
    const Eigen::VectorXd ys = truncated_power(along_y, 0.);
    Eigen::VectorXd u(basis.num_functions());

    for (int function = 0; function < basis.num_functions(); ++function)
        u(function) = xs(function % count) * ys(function / count);

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

    // The penalty reads the degrees of the space, not of the patch: on the
    // quadratic rectangle, the same map, the third derivatives along the
    // first direction still jump, scaled by h^5
    for (const iguana::Patch<Basis, 2>& patch :
         {rectangle(basis), rectangle(uneven())}) {
        iguana::Assembler<Basis, 2> assembler(space, patch, true);
        const Eigen::Index entries = stiffness_of(assembler).nonZeros();
        assembler.assemble_stiffness(ghost, quadrature);

        // The faces were coupled beforehand, so the pattern holds
        REQUIRE(stiffness_of(assembler).nonZeros() == entries);

        const Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness =
            stiffness_of(assembler);

        REQUIRE((stiffness * u).cwiseAbs().maxCoeff() < 1e-12);
        REQUIRE_THAT(v.dot(stiffness * v), WithinRel(expected, 1e-12));
    }
}

TEST_CASE("A ghost penalty needs coupled faces, simple interior knots and an "
          "affine B-spline patch", "[assembler]")
{
    const Basis basis = cubic_by_quadratic();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const iguana::FunctionSpace<Basis> space(basis,
                                             all_cells(CellType::inside));
    const iguana::QuadratureSkeleton<double, 2> quadrature(
        basis.grid(),
        iguana::ghost_faces(basis.grid(), all_cells(CellType::cut)),
        iguana::GaussLegendre<double, 1>(4));

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::SkeletonGhostPenalty<Basis, 2> ghost(trace, 1.);

    REQUIRE_THROWS_AS((iguana::SkeletonGhostPenalty<Basis, 2>(trace, 0.)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((iguana::SkeletonGhostPenalty<Basis, 2>(trace, 1., -1)),
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

    // A repeated knot of the space, across which the second derivatives jump
    // too, though the patch's knots are simple
    const Basis repeated(TensorGrid<double, 2>(
        {KnotVector<double>(3, {0., 0., 0., 0., .2, .6, .6, 1., 1., 1., 1.}),
         KnotVector<double>(2, {0., 0., 0., .7, 1., 1., 1.})}));
    const iguana::FunctionSpace<Basis> repeated_space(
        repeated, all_cells(CellType::inside));
    iguana::Assembler<Basis, 2> on_repeated(repeated_space, patch, true);

    REQUIRE_THROWS_AS(on_repeated.assemble_stiffness(ghost, quadrature),
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
    const iguana::SkeletonGhostPenalty<Rational, 2> rational_ghost(
        rational_trace, 1.);
    iguana::Assembler<Rational, 2> on_rational(rational_space, rational_patch,
                                               true);

    REQUIRE_THROWS_AS(on_rational.assemble_stiffness(rational_ghost,
                                                     quadrature),
                      std::invalid_argument);
}

TEST_CASE("Skeleton Nitsche conditions with binary weights are Nitsche "
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

    const iguana::QuadratureBoundary<double, 2> surrogate(
        iguana::SurrogateBoundary<double, 2>(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));
    const iguana::BoundaryProjection<double> to_surrogate =
        iguana::project_closest_points(patch, boundary, surrogate.points());

    const iguana::QuadratureSkeleton<double, 2> faces(
        basis.grid(), iguana::jump_faces(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));
    const iguana::BoundaryProjection<double> to_faces =
        iguana::project_closest_points(patch, boundary, faces.points());

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::NitscheCondition<Basis, 2> nitsche(trace, flux, 40.);
    const iguana::SkeletonNitscheCondition<Basis, 2> face(trace, flux, 40.);
    const Eigen::Vector2d b(1., -2.);

    iguana::Assembler<Basis, 2> on_surrogate(space, patch, true);
    on_surrogate.assemble_stiffness(nitsche, surrogate, to_surrogate, 2);
    on_surrogate.assemble_load(
        nitsche, surrogate, to_surrogate, 2,
        linear_data(surrogate.points(), to_surrogate, b));

    iguana::Assembler<Basis, 2> on_faces(space, patch, true);
    on_faces.assemble_stiffness(face, faces, to_faces, 2, cell_weights);
    on_faces.assemble_load(face, faces, to_faces, 2,
                           linear_data(faces.points(), to_faces, b),
                           cell_weights);

    // The jump selects the inside cell of every face of the surrogate
    // boundary, whichever side it lies on, and with it its normal
    const Eigen::MatrixXd stiffness = stiffness_of(on_surrogate).toDense();
    const Eigen::MatrixXd difference =
        (stiffness_of(on_surrogate) - stiffness_of(on_faces)).toDense();

    REQUIRE(difference.cwiseAbs().maxCoeff()
            < 1e-12 * stiffness.cwiseAbs().maxCoeff());
    REQUIRE((load_of(on_surrogate) - load_of(on_faces)).cwiseAbs().maxCoeff()
            < 1e-12 * load_of(on_surrogate).cwiseAbs().maxCoeff());
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

    iguana::QuadratureDomain<double, 2> cells;

    for (const CellType type :
         {CellType::inside, CellType::cut, CellType::outside})
        cells.fill(basis.grid(), classification, type,
                   iguana::GaussLegendre<double, 2>(3));

    const iguana::QuadratureSkeleton<double, 2> faces(
        basis.grid(), iguana::jump_faces(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));
    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(patch, boundary, faces.points());

    const iguana::QuadratureSkeleton<double, 2> ghost_faces(
        basis.grid(), iguana::ghost_faces(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));

    const iguana::PoissonElement<Basis, 2> element;
    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::SkeletonNitscheCondition<Basis, 2> face(trace, flux, 40.);
    const iguana::SkeletonGhostPenalty<Basis, 2> ghost(trace, .1);
    const Eigen::Vector2d b(1., -2.);

    iguana::Assembler<Basis, 2> assembler(space, patch, true);
    assembler.assemble_stiffness(element, cells, fractions);
    assembler.assemble_load(element, cells,
                            Eigen::VectorXd::Zero(cells.num_points()),
                            fractions);
    assembler.assemble_stiffness(face, faces, projection, 2, fractions);
    assembler.assemble_load(face, faces, projection, 2,
                            linear_data(faces.points(), projection, b),
                            fractions);
    assembler.assemble_stiffness(ghost, ghost_faces);

    // Cell by cell, the weighted volume terms of b . x leave the jumps of
    // the weights times its flux on the faces, which the condition cancels,
    // and its expansions reach the data exactly. With every cell inside,
    // each function keeps its own index
    const Eigen::VectorXd u = patch.coefficients() * b;

    REQUIRE((stiffness_of(assembler) * u - load_of(assembler)).cwiseAbs()
                .maxCoeff()
            < 1e-12 * load_of(assembler).cwiseAbs().maxCoeff());
}

TEST_CASE("A ghost penalty takes extra powers of the size of the cells and "
          "either sign", "[assembler]")
{
    // Cells of one size h, of sides .4 and .6 in physical space
    const Basis basis = five_by_five();
    const iguana::Patch<Basis, 2> patch = rectangle(basis);
    const std::vector<CellType> types(25, CellType::inside);
    const iguana::FunctionSpace<Basis> space(
        basis, CellClassification<double, 2>(types));
    const iguana::QuadratureSkeleton<double, 2> quadrature(
        basis.grid(),
        iguana::ghost_faces(basis.grid(),
                            CellClassification<double, 2>(
                                std::vector<CellType>(25, CellType::cut))),
        iguana::GaussLegendre<double, 1>(3));
    const iguana::PoissonElement<Basis, 2>::U trace;

    const auto stiffness = [&](double penalty, int scaling) {
        const iguana::SkeletonGhostPenalty<Basis, 2> ghost(trace, penalty,
                                                           scaling);
        iguana::Assembler<Basis, 2> assembler(space, patch, true);
        assembler.assemble_stiffness(ghost, quadrature);

        return Eigen::MatrixXd(stiffness_of(assembler));
    };

    const Eigen::MatrixXd base = stiffness(.3, 0);
    const double h_squared = .2 * width * .2 * height;

    REQUIRE((stiffness(.3, 2) - h_squared * base).norm()
            < 1e-12 * base.norm());
    REQUIRE((stiffness(-.3, 0) + base).norm() == 0.);
}

TEST_CASE("The generalized shifted boundary method holds a divergence-free "
          "quadratic flow exactly", "[assembler]")
{
    using Element = iguana::StokesStabilizedElement<Basis, 2>;

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
        basis,
        CellClassification<double, 2>(std::vector<CellType>(
            static_cast<std::size_t>(count), CellType::inside)),
        3);

    iguana::QuadratureDomain<double, 2> cells;

    for (const CellType type :
         {CellType::inside, CellType::cut, CellType::outside})
        cells.fill(basis.grid(), classification, type,
                   iguana::GaussLegendre<double, 2>(3));

    const iguana::QuadratureSkeleton<double, 2> faces(
        basis.grid(), iguana::jump_faces(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));
    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(patch, boundary, faces.points());
    const iguana::QuadratureSkeleton<double, 2> ghost_faces(
        basis.grid(), iguana::ghost_faces(basis.grid(), classification),
        iguana::GaussLegendre<double, 1>(3));

    const double viscosity = 1.5;
    const Element element(viscosity, 3.);
    const std::array<Element::U, 2> traces{Element::U(0), Element::U(1)};
    const std::array<Element::T, 2> fluxes{Element::T(viscosity, 0),
                                           Element::T(viscosity, 1)};
    const Element::P pressure;

    // f = −div(2μ ε(u)) + ∇p = (1 − 2μ, 1) for the quadratic flow
    Eigen::MatrixXd source(cells.num_points(), 2);
    source.col(0).setConstant(1. - 2. * viscosity);
    source.col(1).setConstant(1.);

    // The velocity at the closest points, which a projection gives in
    // parameter space, where the map scales each axis by its side
    const Eigen::MatrixXd closest = faces.points() + projection.distances;
    const Eigen::VectorXd x = width * closest.col(0);
    const Eigen::VectorXd y = height * closest.col(1);
    const std::array<Eigen::VectorXd, 2> data{x.cwiseProduct(x),
                                              -2. * x.cwiseProduct(y)};

    iguana::Assembler<Basis, 2> assembler(space, patch, true);
    assembler.assemble_stiffness(element, cells, fractions);
    assembler.assemble_load(element, cells, source, fractions);

    for (std::size_t component = 0; component < 2; ++component) {
        const iguana::SkeletonNitscheCondition<Basis, 2> face(
            traces[component], fluxes[component], 40. * viscosity);
        const iguana::SkeletonGhostPenalty<Basis, 2> ghost(
            traces[component], .1 * viscosity);

        assembler.assemble_stiffness(face, faces, projection, 2, fractions);
        assembler.assemble_load(face, faces, projection, 2, data[component],
                                fractions);
        assembler.assemble_stiffness(ghost, ghost_faces);
    }

    // Negative, as the pressure block of the symmetric form is
    const iguana::SkeletonGhostPenalty<Basis, 2> pressure_ghost(
        pressure, -.1 / viscosity, 2);
    assembler.assemble_stiffness(pressure_ghost, ghost_faces);

    // Cell by cell, the weighted volume terms leave the jumps of the
    // weights times the traction on the faces, which the conditions
    // cancel, the expansions of order two reach the data exactly, and the
    // flow, one polynomial, does not jump
    REQUIRE((stiffness_of(assembler) * quadratic_flow(patch)
             - load_of(assembler)).cwiseAbs().maxCoeff()
            < 1e-11 * load_of(assembler).cwiseAbs().maxCoeff());
}

TEST_CASE("Skeleton conditions need coupled faces, one weight per cell and a "
          "projection of their points", "[assembler]")
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

    const iguana::QuadratureSkeleton<double, 2> faces(
        basis.grid(),
        iguana::jump_faces(
            basis.grid(),
            CellClassification<double, 2>(iguana::cell_types(fractions))),
        iguana::GaussLegendre<double, 1>(3));
    const iguana::BoundaryProjection<double> projection =
        iguana::project_closest_points(patch, boundary, faces.points());

    const iguana::PoissonElement<Basis, 2>::U trace;
    const iguana::PoissonElement<Basis, 2>::Q flux;
    const iguana::SkeletonNitscheCondition<Basis, 2> face(trace, flux, 40.);
    const Eigen::VectorXd data = Eigen::VectorXd::Zero(faces.num_points());

    REQUIRE_THROWS_AS(
        (iguana::SkeletonNitscheCondition<Basis, 2>(trace, flux, 0.)),
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

    // A negative order, and a projection of other points
    const iguana::BoundaryProjection<double> other =
        iguana::project_closest_points(patch, boundary,
                                       faces.points().topRows(1));

    REQUIRE_THROWS_AS(assembler.assemble_stiffness(face, faces, projection,
                                                   -1, fractions),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(assembler.assemble_stiffness(face, faces, other, 2,
                                                   fractions),
                      std::invalid_argument);

    // Shift weights for the six multi-indices of order 2, given another
    // order or one row short of the points
    iguana::BoundaryProjection<double> weighted = projection;
    weighted.shift_weights = Eigen::MatrixXd::Zero(faces.num_points(), 6);

    REQUIRE_NOTHROW(assembler.assemble_stiffness(face, faces, weighted, 2,
                                                 fractions));
    REQUIRE_THROWS_AS(assembler.assemble_stiffness(face, faces, weighted, 3,
                                                   fractions),
                      std::invalid_argument);

    weighted.shift_weights.conservativeResize(faces.num_points() - 1, 6);

    REQUIRE_THROWS_AS(assembler.assemble_stiffness(face, faces, weighted, 2,
                                                   fractions),
                      std::invalid_argument);

    // A NURBS patch has no Taylor series for a projection
    using Rational = iguana::TensorNURBS<double, 2>;
    const Rational rational(basis,
                            Eigen::VectorXd::Ones(basis.num_functions()));
    const iguana::Patch<Rational, 2> rational_patch(rational,
                                                    patch.coefficients());
    const iguana::FunctionSpace<Rational> rational_space(
        rational, CellClassification<double, 2>(every));
    const iguana::PoissonElement<Rational, 2>::U rational_trace;
    const iguana::PoissonElement<Rational, 2>::Q rational_flux;
    const iguana::SkeletonNitscheCondition<Rational, 2> rational_face(
        rational_trace, rational_flux, 40.);
    iguana::Assembler<Rational, 2> on_rational(rational_space, rational_patch,
                                               true);

    REQUIRE_THROWS_AS(on_rational.assemble_stiffness(rational_face, faces,
                                                     projection, 2,
                                                     fractions),
                      std::invalid_argument);
}

TEST_CASE("A term on one field of a mixed space is the term of that field "
          "alone, at its degrees of freedom", "[assembler]")
{
    // Three fields of their own degrees on the elements of the quadratic
    // patch, with 24, 25 and 20 degrees of freedom
    const iguana::Patch<Basis, 2> patch = rectangle(uneven());
    const std::array<iguana::FunctionSpace<Basis>, 3> spaces{
        iguana::FunctionSpace<Basis>(cubic_by_quadratic(),
                                     all_cells(CellType::inside)),
        iguana::FunctionSpace<Basis>(quadratic_by_cubic(),
                                     all_cells(CellType::inside)),
        iguana::FunctionSpace<Basis>(uneven(), all_cells(CellType::inside))};
    const iguana::MixedFunctionSpace<Basis> mixed(
        {spaces[0], spaces[1], spaces[2]});
    const std::array<int, 3> offsets{0, 24, 49};

    // A boundary crossing the knot lines, and every interior face
    const iguana::QuadratureBoundary<double, 2> boundary =
        boundary_quadrature(patch, quadrilateral());
    const iguana::QuadratureSkeleton<double, 2> skeleton(
        patch.basis().grid(),
        iguana::ghost_faces(patch.basis().grid(), all_cells(CellType::cut)),
        iguana::GaussLegendre<double, 1>(4));
    const Eigen::VectorXd data = width * boundary.points().col(0)
                                 - 2. * height * boundary.points().col(1);

    iguana::Assembler<Basis, 2> assembler(mixed, patch, true);
    const iguana::PoissonElement<Basis, 2>::U trace;

    for (std::size_t field = 0; field < 3; ++field) {
        INFO("field " << field);
        const iguana::FunctionSpace<Basis>& space = spaces[field];
        const Eigen::MatrixXd place =
            placement(69, offsets[field], space.num_dofs());
        const FieldTrace field_trace(field);
        iguana::Assembler<Basis, 2> alone(space, patch, true);

        // A penalty and its data
        const iguana::PenaltyCondition<Basis, 2> penalty(field_trace, 10.);
        const iguana::PenaltyCondition<Basis, 2> alone_penalty(trace, 10.);
        assembler.clear();
        assembler.assemble_stiffness(penalty, boundary);
        assembler.assemble_load(penalty, boundary, data);
        alone.assemble_stiffness(alone_penalty, boundary);
        alone.assemble_load(alone_penalty, boundary, data);

        REQUIRE(near(stiffness_of(assembler).toDense(),
                     place * stiffness_of(alone).toDense()
                         * place.transpose()));
        REQUIRE(near(load_of(assembler), place * load_of(alone)));

        // A ghost penalty, to the degrees of the field across the faces
        const iguana::SkeletonGhostPenalty<Basis, 2> ghost(field_trace, .3);
        const iguana::SkeletonGhostPenalty<Basis, 2> alone_ghost(trace, .3);
        assembler.clear();
        alone.clear();
        assembler.assemble_stiffness(ghost, skeleton);
        alone.assemble_stiffness(alone_ghost, skeleton);

        REQUIRE(near(stiffness_of(assembler).toDense(),
                     place * stiffness_of(alone).toDense()
                         * place.transpose()));
    }
}

TEST_CASE("Several fields share no block size, and a ghost penalty needs "
          "simple interior knots of its own field", "[assembler]")
{
    const iguana::Patch<Basis, 2> patch = rectangle(uneven());

    // A field of two components followed by one of one, whose degrees of
    // freedom come in no blocks
    const iguana::FunctionSpace<Basis> vector(uneven(),
                                              all_cells(CellType::inside), 2);
    const iguana::FunctionSpace<Basis> scalar(uneven(),
                                              all_cells(CellType::inside));
    const iguana::Assembler<Basis, 2> blocks(
        iguana::MixedFunctionSpace<Basis>({vector, scalar}), patch);
    PetscInt block_size = 0;
    check(MatGetBlockSize(blocks.stiffness(), &block_size));

    REQUIRE(block_size == 1);

    // A first field with a repeated knot and a second with simple ones
    const Basis repeated(TensorGrid<double, 2>(
        {KnotVector<double>(3, {0., 0., 0., 0., .2, .6, .6, 1., 1., 1., 1.}),
         KnotVector<double>(2, {0., 0., 0., .7, 1., 1., 1.})}));
    const iguana::FunctionSpace<Basis> repeated_space(
        repeated, all_cells(CellType::inside));
    iguana::Assembler<Basis, 2> assembler(
        iguana::MixedFunctionSpace<Basis>({repeated_space, scalar}), patch,
        true);
    const iguana::QuadratureSkeleton<double, 2> skeleton(
        patch.basis().grid(),
        iguana::ghost_faces(patch.basis().grid(), all_cells(CellType::cut)),
        iguana::GaussLegendre<double, 1>(4));

    const FieldTrace first(0);
    const FieldTrace second(1);
    const FieldTrace missing(2);
    const iguana::SkeletonGhostPenalty<Basis, 2> on_first(first, 1.);
    const iguana::SkeletonGhostPenalty<Basis, 2> on_second(second, 1.);
    const iguana::SkeletonGhostPenalty<Basis, 2> on_missing(missing, 1.);

    REQUIRE_NOTHROW(assembler.assemble_stiffness(on_second, skeleton));
    REQUIRE_THROWS_AS(assembler.assemble_stiffness(on_first, skeleton),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(assembler.assemble_stiffness(on_missing, skeleton),
                      std::invalid_argument);
}
