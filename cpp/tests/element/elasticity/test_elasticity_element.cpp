/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;
using iguana::KnotVector;

template<std::size_t n>
using Basis = iguana::TensorBSpline<double, n>;

template<std::size_t n>
using Element = iguana::ElasticityElement<Basis<n>, n>;

/// @brief First Lamé parameter and shear modulus of the elements
constexpr double lame = 1.2;
constexpr double shear_modulus = .7;

/// @brief Sides of the box, unequal so that physical and parametric
///        derivatives differ
constexpr std::array<double, 3> sides{2., 3., .5};

/// @brief Displacement gradients of the linear fields u = A x and v = B x
const Eigen::Matrix3d full_a{{1., -2., .3}, {3., .5, -1.}, {.2, 1.5, -.7}};
const Eigen::Matrix3d full_b{{-1., 2., .4}, {.5, 4., -.2}, {1.1, -.3, .9}};

/// @brief The Bernstein polynomials of degree two along each direction
template<std::size_t n>
std::array<KnotVector<double>, n> bernstein()
{
    const KnotVector<double> knots(2, {0., 0., 0., 1., 1., 1.});

    if constexpr (n == 2)
        return {knots, knots};
    else if constexpr (n == 3)
        return {knots, knots, knots};
}

/// @brief Box of one element, whose control points are the scaled Greville
///        points, so that the map scales each axis by its side
template<std::size_t n>
iguana::Patch<Basis<n>, n> box()
{
    const Basis<n> basis{iguana::TensorGrid<double, n>(bernstein<n>())};
    std::array<int, n> counts{};
    counts.fill(3);

    iguana::PointMatrix<double, n> points(basis.num_functions(), n);

    for (int function = 0; function < basis.num_functions(); ++function) {
        const std::array<int, n> index = iguana::unflatten(function, counts);

        for (std::size_t direction = 0; direction < n; ++direction)
            points(function, direction) =
                sides[direction] * index[direction] / 2.;
    }

    return {basis, points};
}

/// @brief Local coefficients of the field A x + t, on a patch whose control
///        points reproduce x, one component after another
template<std::size_t n>
Eigen::VectorXd field(const iguana::Patch<Basis<n>, n>& patch,
                      const Eigen::MatrixXd& a, const Eigen::VectorXd& t)
{
    Eigen::MatrixXd values = patch.coefficients() * a.transpose();
    values.rowwise() += t.transpose();

    return Eigen::Map<const Eigen::VectorXd>(values.data(), values.size());
}

/// @brief Whether a matrix is zero, up to rounding relative to a scale
bool vanishes(const Eigen::MatrixXd& matrix, double scale)
{
    return matrix.cwiseAbs().maxCoeff() <= 1e-12 * scale;
}

/**
 * @brief Runs a check on the values of a box at Gauss points, which integrate
 *        the products of its functions exactly
 *
 * @param check Check of the patch, the values, the points in parameter space
 *        and the physical weights
 */
template<std::size_t n, typename Check>
void on_box(const Check& check)
{
    const iguana::Patch<Basis<n>, n> patch = box<n>();
    iguana::ElementValues<Basis<n>, n> values(
        patch, iguana::ValueFlags{.physical_gradients = true});

    std::array<double, n> start{};
    std::array<double, n> end{};
    end.fill(1.);

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    iguana::GaussLegendre<double, n>(3).fill_to_reference_space(
        start, end, points, weights);
    iguana::BoxRule<double, n>::map_to_parameter_space(start, end, points,
                                                       weights);
    values.reinit(0, points);
    weights = weights.cwiseProduct(values.measures());

    check(patch, values, points, weights);
}

/// @brief Checks the stiffness, residual and mass of an element against
///        their exact integrals
template<std::size_t n>
void check_forms()
{
    on_box<n>([](const auto& patch, const auto& values, const auto& points,
                 const auto& weights) {
        const Element<n> element(lame, shear_modulus);
        const Eigen::MatrixXd a = full_a.topLeftCorner(n, n);
        const Eigen::MatrixXd b = full_b.topLeftCorner(n, n);
        const Eigen::VectorXd zero_shift = Eigen::VectorXd::Zero(n);
        const Eigen::VectorXd u = field<n>(patch, a, zero_shift);
        const Eigen::VectorXd v = field<n>(patch, b, zero_shift);
        const Eigen::VectorXd zero = Eigen::VectorXd::Zero(u.size());

        double volume = 1.;

        for (std::size_t direction = 0; direction < n; ++direction)
            volume *= sides[direction];

        // A constant body force
        const Eigen::VectorXd force = Eigen::VectorXd::LinSpaced(n, 1., 2.);
        const Eigen::MatrixXd source =
            force.transpose().replicate(points.rows(), 1);

        Eigen::MatrixXd stiffness;
        element.local_tangent(values, weights, source, zero, stiffness);

        REQUIRE(stiffness.rows() == u.size());
        REQUIRE(stiffness.cols() == u.size());

        // ( 2μ ε(u), ε(v) )_Ω + ( λ div u, div v )_Ω of the linear fields
        const Eigen::MatrixXd strain_a = (a + a.transpose()) / 2.;
        const Eigen::MatrixXd strain_b = (b + b.transpose()) / 2.;
        const double energy =
            volume * (2. * shear_modulus
                          * strain_a.cwiseProduct(strain_b).sum()
                      + lame * a.trace() * b.trace());

        REQUIRE_THAT(v.dot(stiffness * u), WithinRel(energy, 1e-12));

        // Rigid motions store no energy, whatever the other functions do
        const Eigen::MatrixXd skew = a - a.transpose();
        const double scale = stiffness.cwiseAbs().maxCoeff();

        REQUIRE(vanishes(stiffness * field<n>(patch, skew, force), scale));
        REQUIRE(vanishes(stiffness - stiffness.transpose(), scale));

        // At rest, the residual is minus the load, which a constant field
        // tests against the force
        const Eigen::MatrixXd no_gradient = Eigen::MatrixXd::Zero(n, n);
        const Eigen::VectorXd one = Eigen::VectorXd::LinSpaced(n, 3., -1.);
        const Eigen::VectorXd constant = field<n>(patch, no_gradient, one);
        Eigen::VectorXd at_zero;
        element.local_residual(values, weights, source, zero, at_zero);

        REQUIRE_THAT(-constant.dot(at_zero),
                     WithinRel(volume * force.dot(one), 1e-12));

        // The mass of unit density of two constant fields
        const Eigen::VectorXd other =
            field<n>(patch, no_gradient, force);
        Eigen::MatrixXd mass;
        element.local_mass(values, weights, mass);

        REQUIRE_THAT(constant.dot(mass * other),
                     WithinRel(volume * force.dot(one), 1e-12));
    });
}

/// @brief Checks that the traces and fluxes give the displacement A x and
///        the traction (λ tr(A) I + μ (A + A^T)) n
template<std::size_t n>
void check_traces()
{
    using U = typename Element<n>::U;
    using T = typename Element<n>::T;

    on_box<n>([](const auto& patch, const auto& values, const auto& points,
                 const auto&) {
        const Eigen::MatrixXd a = full_a.topLeftCorner(n, n);
        const Eigen::VectorXd u =
            field<n>(patch, a, Eigen::VectorXd::Zero(n));

        // The points in physical space, and one normal at each
        Eigen::MatrixXd positions = points;

        for (std::size_t direction = 0; direction < n; ++direction) {
            const Eigen::Index column = static_cast<Eigen::Index>(direction);
            positions.col(column) *= sides[direction];
        }

        const Eigen::VectorXd normal =
            Eigen::VectorXd::LinSpaced(n, 1., 2.).normalized();
        const iguana::PointMatrix<double, n> normals =
            normal.transpose().replicate(points.rows(), 1);
        const Eigen::VectorXd traction =
            (lame * a.trace() * Eigen::MatrixXd::Identity(n, n)
             + shear_modulus * (a + a.transpose()))
            * normal;

        Eigen::MatrixXd trace;
        Eigen::MatrixXd flux;

        for (std::size_t c = 0; c < n; ++c) {
            INFO("component " << c);
            const Eigen::Index component = static_cast<Eigen::Index>(c);

            U(c).local_trace(values, trace);
            T(lame, shear_modulus, c).local_flux(values, normals, flux);

            REQUIRE(trace.rows() == u.size());
            REQUIRE(flux.rows() == u.size());
            REQUIRE(vanishes(trace.transpose() * u
                                 - positions * a.row(component).transpose(),
                             1.));
            REQUIRE(((flux.transpose() * u).array() - traction(component))
                        .abs()
                        .maxCoeff() < 1e-12);
            REQUIRE(U(c).field(1) == 0);
        }
    });
}

/// @brief Checks the moduli and components an element and its traces reject
template<std::size_t n>
void check_invalid()
{
    using U = typename Element<n>::U;
    using T = typename Element<n>::T;

    REQUIRE_THROWS_AS(Element<n>(1., 0.), std::invalid_argument);
    REQUIRE_THROWS_AS(T(1., 0., 0), std::invalid_argument);

    // A negative λ is fine as long as the bulk modulus stays positive
    REQUIRE_NOTHROW(Element<n>(-.1, 1.));
    REQUIRE_THROWS_AS(Element<n>(-1., .1), std::invalid_argument);
    REQUIRE_THROWS_AS(T(-1., .1, 0), std::invalid_argument);

    REQUIRE_THROWS_AS(U(n), std::invalid_argument);
    REQUIRE_THROWS_AS(T(1., 1., n), std::invalid_argument);
}

} // namespace

TEST_CASE("The forms of a linear elasticity element are exact and leave "
          "rigid motions free", "[elasticity_element]")
{
    check_forms<2>();
    check_forms<3>();
}

TEST_CASE("The traces and fluxes of linear elasticity give the displacement "
          "and the traction", "[elasticity_element]")
{
    check_traces<2>();
    check_traces<3>();
}

TEST_CASE("A linear elasticity element needs a positive shear and bulk "
          "modulus, and its traces a component of the displacement",
          "[elasticity_element]")
{
    check_invalid<2>();
    check_invalid<3>();
}
