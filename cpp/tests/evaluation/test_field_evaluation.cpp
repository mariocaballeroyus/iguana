/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

namespace
{

using iguana::CellClassification;
using iguana::CellType;
using iguana::FunctionSpace;
using iguana::MixedFunctionSpace;
using iguana::Patch;
using iguana::PointMatrix;

template<std::size_t n>
using Basis = iguana::TensorBSpline<double, n>;

/// @brief Dynamic viscosity of the flow
constexpr double viscosity = .7;

/// @brief Sheared map of the patches, x = a + J ξ, so that physical and
///        parametric derivatives differ
const Eigen::Matrix3d full_map{{2., .5, .1}, {-.75, 1.5, .2}, {.3, -.4, 1.}};
const Eigen::Vector3d full_offset(.5, -1., .25);

/// @brief Linear flow, u = A x + b and p = c · x + d
const Eigen::Matrix3d full_a{{1., -2., .3}, {3., .5, -1.}, {.2, 1.5, -.7}};
const Eigen::Vector3d full_b(.4, -1.1, .6);
const Eigen::Vector3d full_c(-.5, 2., .9);
constexpr double full_d = 1.3;

/// @brief Whether two arrays agree, up to rounding
bool agree(const Eigen::MatrixXd& computed, const Eigen::MatrixXd& exact)
{
    return (computed - exact).cwiseAbs().maxCoeff() <= 1e-11;
}

/// @brief B-splines of some degrees with interior knots along each direction
template<std::size_t n>
Basis<n> spline(const std::array<int, n>& degrees)
{
    const std::array<std::vector<double>, 3> interior{{{.3, .7}, {.4}, {.6}}};

    const auto axis = [&degrees, &interior](std::size_t direction) {
        const std::vector<double>& lines = interior[direction];
        const int degree = degrees[direction];

        std::vector<double> knots(degree + 1, 0.);
        knots.insert(knots.end(), lines.begin(), lines.end());
        knots.insert(knots.end(), degree + 1, 1.);

        return iguana::BSpline<double>(degree, knots);
    };

    if constexpr (n == 2)
        return Basis<n>({axis(0), axis(1)});
    else if constexpr (n == 3)
        return Basis<n>({axis(0), axis(1), axis(2)});
}

/// @brief Degrees of the basis of the patches
template<std::size_t n>
std::array<int, n> base_degrees()
{
    if constexpr (n == 2)
        return {2, 3};
    else if constexpr (n == 3)
        return {2, 3, 2};
}

/// @brief Images under the map of parameters, one per row
template<std::size_t n>
PointMatrix<double, n> mapped(const Eigen::MatrixXd& parameters)
{
    const Eigen::MatrixXd linear = full_map.topLeftCorner(n, n);
    const Eigen::RowVectorXd offset = full_offset.head(n).transpose();

    PointMatrix<double, n> points = parameters * linear.transpose();
    points.rowwise() += offset;

    return points;
}

/// @brief Images of the Greville points of a basis, one per function. An
///        affine function of x has its values there as coefficients
template<std::size_t n>
PointMatrix<double, n> greville_images(const Basis<n>& basis)
{
    std::array<Eigen::VectorXd, n> abscissae;
    std::array<int, n> counts{};

    for (std::size_t direction = 0; direction < n; ++direction) {
        abscissae[direction] =
            iguana::greville_abscissae(basis.axis(direction).knots());
        counts[direction] = static_cast<int>(abscissae[direction].size());
    }

    Eigen::MatrixXd parameters(basis.num_functions(), n);

    for (int function = 0; function < basis.num_functions(); ++function) {
        const std::array<int, n> index = iguana::unflatten(function, counts);

        for (std::size_t direction = 0; direction < n; ++direction)
            parameters(function, direction) =
                abscissae[direction](index[direction]);
    }

    return mapped<n>(parameters);
}

/// @brief Points inside elements, on every knot line and at the corners
template<std::size_t n>
PointMatrix<double, n> sample_points()
{
    const std::vector<double> values{0., .2, .3, .4, .6, .7, .85, 1.};
    std::array<int, n> counts{};
    counts.fill(static_cast<int>(values.size()));

    int num_points = 1;

    for (const int count : counts)
        num_points *= count;

    Eigen::MatrixXd parameters(num_points, n);

    for (int point = 0; point < num_points; ++point) {
        const std::array<int, n> index = iguana::unflatten(point, counts);

        for (std::size_t direction = 0; direction < n; ++direction)
            parameters(point, direction) =
                values[static_cast<std::size_t>(index[direction])];
    }

    return mapped<n>(parameters);
}

/// @brief Velocity of the flow at points, one component per column
template<std::size_t n>
Eigen::MatrixXd velocity(const PointMatrix<double, n>& points)
{
    const Eigen::MatrixXd a = full_a.topLeftCorner(n, n);
    const Eigen::RowVectorXd b = full_b.head(n).transpose();

    Eigen::MatrixXd values = points * a.transpose();
    values.rowwise() += b;

    return values;
}

/// @brief Pressure of the flow at points
template<std::size_t n>
Eigen::VectorXd pressure(const PointMatrix<double, n>& points)
{
    const Eigen::VectorXd c = full_c.head(n);

    return (points * c).array() + full_d;
}

/// @brief Every cell of a basis inside the physical domain
template<std::size_t n>
CellClassification<double, n> all_inside(int num_elements)
{
    return CellClassification<double, n>(std::vector<CellType>(
        static_cast<std::size_t>(num_elements), CellType::inside));
}

/**
 * @brief Coefficients of the flow on one field of a basis, given the images
 *        of its Greville points, the components of a function being adjacent
 */
template<std::size_t n>
Eigen::VectorXd block_coefficients(const PointMatrix<double, n>& images)
{
    Eigen::MatrixXd components(n + 1, images.rows());
    components.topRows(n) = velocity<n>(images).transpose();
    components.row(n) = pressure<n>(images).transpose();

    return components.reshaped();
}

/**
 * @brief Checks the velocity, the pressure and the traction of the flow from
 *        its coefficients on a space, at points and normals of the patch
 */
template<typename BasisType, std::size_t n>
void check_flow(const MixedFunctionSpace<BasisType>& space,
                const Patch<BasisType, n>& patch,
                const Eigen::VectorXd& coefficients,
                const PointMatrix<double, n>& points)
{
    using Element = iguana::StokesElement<BasisType, n>;

    // Unit normals that differ from point to point
    PointMatrix<double, n> normals = points;
    normals.rowwise() += Eigen::RowVectorXd::Constant(n, 7.);
    normals.rowwise().normalize();

    const Eigen::MatrixXd a = full_a.topLeftCorner(n, n);
    const Eigen::MatrixXd exact_velocity = velocity<n>(points);
    const Eigen::VectorXd exact_pressure = pressure<n>(points);

    // t = (μ (A + A^T) − p I) n
    Eigen::MatrixXd exact_traction =
        viscosity * normals * (a + a.transpose());
    exact_traction -= (normals.array().colwise() * exact_pressure.array())
                          .matrix();

    for (std::size_t component = 0; component < n; ++component) {
        const typename Element::U trace(component);
        const typename Element::T flux(viscosity, component);

        CHECK(agree(iguana::trace_at_points(space, patch, trace, coefficients,
                                            points),
                    exact_velocity.col(component)));
        CHECK(agree(iguana::flux_at_points(space, patch, flux, coefficients,
                                           points, normals),
                    exact_traction.col(component)));
    }

    const typename Element::P trace;

    CHECK(agree(iguana::trace_at_points(space, patch, trace, coefficients,
                                        points),
                exact_pressure));
}

/// @brief Checks the flow on one field of a basis, its components those of
///        the block
template<typename BasisType, std::size_t n>
void check_one_field(const BasisType& basis,
                     const PointMatrix<double, n>& images)
{
    const FunctionSpace<BasisType> field(
        basis, all_inside<n>(basis.grid().num_elements()),
        static_cast<int>(n) + 1);
    const MixedFunctionSpace<BasisType> space(std::vector{std::cref(field)});
    const Patch<BasisType, n> patch(basis, images);

    check_flow<BasisType, n>(space, patch, block_coefficients<n>(images),
                         sample_points<n>());
}

/// @brief Checks the flow on a field per unknown, each component of the
///        velocity one degree higher along its own direction
template<std::size_t n>
void check_several_fields()
{
    const Basis<n> basis = spline<n>(base_degrees<n>());
    const CellClassification<double, n> cells =
        all_inside<n>(basis.grid().num_elements());

    std::vector<FunctionSpace<Basis<n>>> fields;
    fields.reserve(n + 1);

    for (std::size_t component = 0; component < n; ++component) {
        std::array<int, n> degrees = base_degrees<n>();
        ++degrees[component];
        fields.emplace_back(spline<n>(degrees), cells);
    }

    fields.emplace_back(basis, cells);

    std::vector<std::reference_wrapper<const FunctionSpace<Basis<n>>>> views(
        fields.begin(), fields.end());
    const MixedFunctionSpace<Basis<n>> space(views);

    // One field after another
    Eigen::VectorXd coefficients(space.num_dofs());

    for (std::size_t component = 0; component < n; ++component) {
        const FunctionSpace<Basis<n>>& field = fields[component];
        const PointMatrix<double, n> images = greville_images<n>(field.basis());

        coefficients.segment(space.offset(component), images.rows()) =
            velocity<n>(images).col(component);
    }

    const PointMatrix<double, n> images = greville_images<n>(basis);

    coefficients.segment(space.offset(n), images.rows()) =
        pressure<n>(images);

    check_flow<Basis<n>, n>(space, Patch<Basis<n>, n>(basis, images),
                            coefficients, sample_points<n>());
}

/// @brief Checks the flow on one field of B-splines, and of NURBS with equal
///        weights, whose map is still affine
template<std::size_t n>
void check_one_field_bases()
{
    const Basis<n> basis = spline<n>(base_degrees<n>());
    const PointMatrix<double, n> images = greville_images<n>(basis);

    check_one_field<Basis<n>, n>(basis, images);

    const iguana::TensorNURBS<double, n> rational(
        basis, Eigen::VectorXd::Constant(basis.num_functions(), 2.));

    check_one_field<iguana::TensorNURBS<double, n>, n>(rational, images);
}

} // namespace

TEST_CASE("Field evaluation gives the velocity, pressure and traction of a "
          "flow in elements, on knot lines and at corners",
          "[field_evaluation]")
{
    SECTION("On one field")
    {
        check_one_field_bases<2>();
        check_one_field_bases<3>();
    }

    SECTION("On a field per unknown")
    {
        check_several_fields<2>();
        check_several_fields<3>();
    }
}

TEST_CASE("Field evaluation reads the functions a space keeps and rejects "
          "the cells outside it", "[field_evaluation]")
{
    using Element = iguana::StokesElement<Basis<2>, 2>;

    const Basis<2> basis = spline<2>(base_degrees<2>());
    const PointMatrix<double, 2> images = greville_images<2>(basis);
    const Patch<Basis<2>, 2> patch(basis, images);

    // The first cell alone supports the first function, so that each degree
    // of freedom belongs to the function after its own index
    std::vector<CellType> types(
        static_cast<std::size_t>(basis.grid().num_elements()),
        CellType::inside);
    types.front() = CellType::outside;

    const FunctionSpace<Basis<2>> field(
        basis, CellClassification<double, 2>(types), 3);
    const MixedFunctionSpace<Basis<2>> space(std::vector{std::cref(field)});

    REQUIRE(field.dof_map().num_dofs() == basis.num_functions() - 1);

    const PointMatrix<double, 2> kept = images.bottomRows(images.rows() - 1);
    const Eigen::VectorXd coefficients = block_coefficients<2>(kept);

    // Points on the other cells, the first being [0, .3] x [0, .4]
    const Eigen::MatrixXd parameters{{.3, .2}, {.1, .4}, {.5, .1}, {1., 1.}};

    check_flow<Basis<2>, 2>(space, patch, coefficients,
                            mapped<2>(parameters));

    const Eigen::MatrixXd inside_first{{.1, .2}};

    CHECK_THROWS_AS(iguana::trace_at_points(space, patch, Element::P(),
                                            coefficients,
                                            mapped<2>(inside_first)),
                    std::invalid_argument);
}

TEST_CASE("Field evaluation takes points on the edge of a patch up to "
          "round-off, and no further", "[field_evaluation]")
{
    using Element = iguana::StokesElement<Basis<2>, 2>;

    const Basis<2> basis = spline<2>(base_degrees<2>());
    const PointMatrix<double, 2> images = greville_images<2>(basis);
    const FunctionSpace<Basis<2>> field(
        basis, all_inside<2>(basis.grid().num_elements()), 3);
    const MixedFunctionSpace<Basis<2>> space(std::vector{std::cref(field)});
    const Patch<Basis<2>, 2> patch(basis, images);
    const Eigen::VectorXd coefficients = block_coefficients<2>(images);

    const auto trace_at = [&](double xi, double eta) {
        const Eigen::MatrixXd parameters{{xi, eta}};

        return iguana::trace_at_points(space, patch, Element::P(),
                                       coefficients, mapped<2>(parameters));
    };

    const Eigen::MatrixXd rounded{{1. + 1e-13, -1e-13}};

    CHECK(agree(trace_at(1. + 1e-13, -1e-13),
                pressure<2>(mapped<2>(rounded))));

    CHECK_THROWS_AS(trace_at(1. + 1e-9, .5), std::invalid_argument);
    CHECK_THROWS_AS(trace_at(.5, -1e-9), std::invalid_argument);
}

TEST_CASE("Field evaluation rejects what defines no value",
          "[field_evaluation]")
{
    using Element = iguana::StokesElement<Basis<2>, 2>;

    const Basis<2> basis = spline<2>(base_degrees<2>());
    const PointMatrix<double, 2> images = greville_images<2>(basis);
    const FunctionSpace<Basis<2>> field(
        basis, all_inside<2>(basis.grid().num_elements()), 3);
    const MixedFunctionSpace<Basis<2>> space(std::vector{std::cref(field)});
    const Patch<Basis<2>, 2> patch(basis, images);
    const Eigen::VectorXd coefficients = block_coefficients<2>(images);
    const PointMatrix<double, 2> points = sample_points<2>();
    const Element::P trace;
    const Element::T flux(viscosity, 0);

    SECTION("Coefficients that are not one per degree of freedom")
    {
        CHECK_THROWS_AS(iguana::trace_at_points(space, patch, trace,
                                                coefficients.head(3), points),
                        std::invalid_argument);
    }

    SECTION("Normals that are not one per point")
    {
        const PointMatrix<double, 2> normals = points.topRows(1);

        CHECK_THROWS_AS(iguana::flux_at_points(space, patch, flux,
                                               coefficients, points, normals),
                        std::invalid_argument);
    }

    SECTION("A trace of an element whose unknowns the space does not hold")
    {
        const iguana::PoissonElement<Basis<2>, 2>::U poisson;

        CHECK_THROWS_AS(iguana::trace_at_points(space, patch, poisson,
                                                coefficients, points),
                        std::invalid_argument);
    }

    SECTION("A map that is not affine")
    {
        PointMatrix<double, 2> moved = images;
        moved(0, 0) -= .1;

        const Patch<Basis<2>, 2> curved(basis, moved);

        CHECK_THROWS_AS(iguana::trace_at_points(space, curved, trace,
                                                coefficients, points),
                        std::invalid_argument);
    }

    SECTION("A space without the elements of the patch")
    {
        const Basis<2> other({iguana::BSpline<double>(1, {0., 0., 1., 1.}),
                              iguana::BSpline<double>(1, {0., 0., 1., 1.})});
        const Patch<Basis<2>, 2> coarse(other, greville_images<2>(other));

        CHECK_THROWS_AS(iguana::trace_at_points(space, coarse, trace,
                                                coefficients, points),
                        std::invalid_argument);
    }
}
