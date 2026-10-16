/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::KnotVector;
using iguana::TensorBSpline;
using iguana::TensorGrid;

/// @brief Quadratic knot vectors of the unit interval over uneven spans,
///        with three, two and four elements
const std::array<KnotVector<double>, 3> knot_vectors{
    KnotVector<double>(2, {0., 0., 0., .2, .6, 1., 1., 1.}),
    KnotVector<double>(2, {0., 0., 0., .7, 1., 1., 1.}),
    KnotVector<double>(2, {0., 0., 0., .1, .3, .6, 1., 1., 1.})};

/// @brief Knot vectors of the first d directions
template<std::size_t d>
std::array<KnotVector<double>, d> first_knot_vectors()
{
    if constexpr (d == 2)
        return {knot_vectors[0], knot_vectors[1]};
    else if constexpr (d == 3)
        return knot_vectors;
}

/// @brief Sides of the box, which differs from the parameter box so that
///        the measure of the map matters
constexpr std::array<double, 3> sides{2., 3., .5};

/**
 * @brief Patch of a box whose interior control points are moved off the
 *        scaled identity, so that the map is curved but the boundary, which
 *        depends on the boundary control points alone, stays that of the box
 *
 * @param bend Amplitude of the moves relative to the sides, zero for the
 *        scaled identity itself
 */
template<std::size_t d>
iguana::Patch<TensorBSpline<double, d>, d> bent_box(double bend = .03)
{
    const std::array<KnotVector<double>, d> knots = first_knot_vectors<d>();
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] =
            static_cast<int>(knots[direction].values().size()) - 3;

    const TensorBSpline<double, d> basis{TensorGrid<double, d>(knots)};
    iguana::PointMatrix<double, d> points(basis.num_functions(), d);

    for (int function = 0; function < basis.num_functions(); ++function) {
        const std::array<int, d> index = iguana::unflatten(function, counts);
        bool interior = true;

        for (std::size_t direction = 0; direction < d; ++direction) {
            // The scaled Greville point reproduces the scaled identity
            const std::vector<double>& t = knots[direction].values();
            const std::size_t i = static_cast<std::size_t>(index[direction]);

            points(function, direction) =
                sides[direction] * (t[i + 1] + t[i + 2]) / 2.;
            interior = interior && index[direction] > 0
                       && index[direction] < counts[direction] - 1;
        }

        if (interior)
            for (std::size_t direction = 0; direction < d; ++direction)
                points(function, direction) +=
                    bend * sides[direction]
                    * std::sin(1. + function * (direction + 1.));
    }

    return {basis, points};
}

/// @brief The bent box of two directions, rotated out of the plane into a
///        flat shell of the same area
iguana::Patch<TensorBSpline<double, 2>, 3> bent_shell()
{
    const iguana::Patch<TensorBSpline<double, 2>, 2> box = bent_box<2>();
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(.7, Eigen::Vector3d(1., 2., 3.).normalized())
            .toRotationMatrix();

    iguana::PointMatrix<double, 3> points =
        iguana::PointMatrix<double, 3>::Zero(box.coefficients().rows(), 3);
    points.leftCols<2>() = box.coefficients();

    const iguana::PointMatrix<double, 3> rotated =
        points * rotation.transpose();

    return {box.basis(), rotated};
}

/**
 * @brief Integrates the energy of the linear field b . x over a bent box:
 *        its gradient is b at every point, and its energy |b|^2 times the
 *        volume of the box
 */
template<std::size_t d>
void check_linear_field()
{
    using Basis = TensorBSpline<double, d>;

    const iguana::Patch<Basis, d> patch = bent_box<d>();
    const Eigen::Vector<double, d> b = Eigen::Vector<double, d>::LinSpaced(
        1., -2.);

    // The polynomial measure has degree 2 d per direction at most
    const iguana::GaussLegendre<double, d> rule(4);
    iguana::ElementValues<Basis, d> element_values(
        patch, {.physical_gradients = true});

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    Eigen::VectorXi actives;
    double box = 1.;
    double volume = 0.;
    double energy = 0.;

    for (std::size_t direction = 0; direction < d; ++direction)
        box *= sides[direction];

    for (const auto& element : patch.basis().grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, d>::map_to_parameter_space(
            element.start(), element.end(), points, weights);
        element_values.reinit(element.index(), points);

        // The coefficients of b . x are b . x_A, paired with the rows by
        // the active functions of the element
        patch.basis().active_on_element(element.index(), actives);
        const Eigen::VectorXd u =
            patch.coefficients()(actives, Eigen::placeholders::all) * b;

        for (Eigen::Index point = 0; point < points.rows(); ++point) {
            INFO("point " << point);
            Eigen::Vector<double, d> gradient;

            for (std::size_t c = 0; c < d; ++c)
                gradient(c) = u.dot(
                    element_values.physical_gradients()[c].col(point));

            REQUIRE((gradient - b).cwiseAbs().maxCoeff() < 1e-12);

            const double weight =
                weights(point) * element_values.measures()(point);
            volume += weight;
            energy += weight * gradient.squaredNorm();
        }
    }

    REQUIRE_THAT(volume, WithinAbs(box, 1e-12));
    REQUIRE_THAT(energy, WithinAbs(b.squaredNorm() * box, 1e-12));
}

/**
 * @brief Checks the sizes of the elements of a bent box: h^d is the measure
 *        times the volume of the element in parameter space, so its mean
 *        over the element in parameter space is the measure of the element
 */
template<std::size_t d>
void check_sizes()
{
    using Basis = TensorBSpline<double, d>;

    const iguana::Patch<Basis, d> patch = bent_box<d>();
    const iguana::GaussLegendre<double, d> rule(4);
    iguana::ElementValues<Basis, d> element_values(patch, {});

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    for (const auto& element : patch.basis().grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, d>::map_to_parameter_space(
            element.start(), element.end(), points, weights);
        element_values.reinit(element.index(), points);

        double volume = 1.;

        for (std::size_t direction = 0; direction < d; ++direction)
            volume *= element.end()[direction] - element.start()[direction];

        const Eigen::VectorXd powers =
            element_values.sizes().array().pow(static_cast<double>(d));
        const double mean = weights.dot(powers) / volume;

        REQUIRE_THAT(mean, WithinAbs(weights.dot(element_values.measures()),
                                     1e-12));
    }
}

/**
 * @brief Coefficients of the quadratic field sum of x_k^2 plus sum of
 *        (a + b) x_a x_b over a < b on the box of the scaled identity,
 *        whose Hessian is 2 on the diagonal and a + b off it
 *
 * In a quadratic direction of the box, the polar forms give 1,
 * (t_{i+1} + t_{i+2}) / 2 and t_{i+1} t_{i+2} as the coefficients of 1, xi
 * and xi^2, and x_k is the side times xi_k
 */
template<std::size_t d>
Eigen::VectorXd quadratic_field(const TensorBSpline<double, d>& basis)
{
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] = basis.axis(direction).num_functions();

    Eigen::VectorXd u(basis.num_functions());

    for (int function = 0; function < basis.num_functions(); ++function) {
        const std::array<int, d> index = iguana::unflatten(function, counts);
        std::array<double, d> linear{};
        std::array<double, d> square{};

        for (std::size_t direction = 0; direction < d; ++direction) {
            const std::vector<double>& t =
                basis.grid().knots(direction).values();
            const std::size_t i = static_cast<std::size_t>(index[direction]);

            linear[direction] = sides[direction] * (t[i + 1] + t[i + 2]) / 2.;
            square[direction] =
                sides[direction] * sides[direction] * t[i + 1] * t[i + 2];
        }

        u(function) = 0.;

        for (std::size_t a = 0; a < d; ++a) {
            u(function) += square[a];

            for (std::size_t b = a + 1; b < d; ++b)
                u(function) += static_cast<double>(a + b) * linear[a]
                               * linear[b];
        }
    }

    return u;
}

/**
 * @brief Checks the physical second derivatives of a quadratic field on the
 *        box of the scaled identity, which the quadratic space holds, in the
 *        order of the pairs: the pure ones, then the mixed ones in
 *        lexicographic order
 */
template<std::size_t d>
void check_quadratic_field()
{
    using Basis = TensorBSpline<double, d>;

    const iguana::Patch<Basis, d> patch = bent_box<d>(0.);
    const Eigen::VectorXd field = quadratic_field<d>(patch.basis());
    const iguana::GaussLegendre<double, d> rule(2);
    iguana::ElementValues<Basis, d> element_values(
        patch, {.physical_hessians = true});

    std::array<double, d * (d + 1) / 2> expected{};
    std::size_t pair = d;

    for (std::size_t a = 0; a < d; ++a) {
        expected[a] = 2.;

        for (std::size_t b = a + 1; b < d; ++b)
            expected[pair++] = static_cast<double>(a + b);
    }

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    Eigen::VectorXi actives;

    for (const auto& element : patch.basis().grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, d>::map_to_parameter_space(
            element.start(), element.end(), points, weights);
        element_values.reinit(element.index(), points);
        patch.basis().active_on_element(element.index(), actives);

        const Eigen::VectorXd u = field(actives);

        for (std::size_t pair_of = 0; pair_of < expected.size(); ++pair_of) {
            INFO("pair " << pair_of);
            const Eigen::RowVectorXd hessian =
                u.transpose() * element_values.physical_hessians()[pair_of];

            REQUIRE((hessian.array() - expected[pair_of]).abs().maxCoeff()
                    < 1e-11);
        }
    }
}

/**
 * @brief Checks that the coordinates x_k, which an isoparametric space
 *        holds whatever the map, have no physical second derivatives on a
 *        curved one, which only the curvature term of the map cancels
 */
template<typename Basis, std::size_t d>
void check_coordinate_fields(const iguana::Patch<Basis, d>& patch)
{
    const iguana::GaussLegendre<double, d> rule(3);
    iguana::ElementValues<Basis, d> element_values(
        patch, {.physical_hessians = true});

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    Eigen::VectorXi actives;

    for (const auto& element : patch.basis().grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, d>::map_to_parameter_space(
            element.start(), element.end(), points, weights);
        element_values.reinit(element.index(), points);
        patch.basis().active_on_element(element.index(), actives);

        const Eigen::MatrixXd coordinates =
            patch.coefficients()(actives, Eigen::placeholders::all);

        for (const Eigen::MatrixXd& hessians :
             element_values.physical_hessians())
            REQUIRE((coordinates.transpose() * hessians).cwiseAbs().maxCoeff()
                    < 1e-10);
    }
}

} // namespace

TEST_CASE("Element values integrate the energy of a linear field exactly",
          "[element_values]")
{
    check_linear_field<2>();
    check_linear_field<3>();
}

TEST_CASE("Element sizes average to the measure of each element",
          "[element_values]")
{
    check_sizes<2>();
    check_sizes<3>();
}

TEST_CASE("Element values give the second derivatives of a quadratic "
          "field in physical space", "[element_values]")
{
    check_quadratic_field<2>();
    check_quadratic_field<3>();
}

TEST_CASE("Element values cancel the curvature of the map in the second "
          "derivatives", "[element_values]")
{
    check_coordinate_fields(bent_box<2>());
    check_coordinate_fields(bent_box<3>());

    // Uneven weights make the functions and the map rational
    const iguana::Patch<TensorBSpline<double, 2>, 2> box = bent_box<2>();
    const Eigen::VectorXd weights =
        Eigen::VectorXd::LinSpaced(box.basis().num_functions(), .6, 1.4);
    const iguana::TensorNURBS<double, 2> rational(box.basis(), weights);

    check_coordinate_fields(
        iguana::Patch<iguana::TensorNURBS<double, 2>, 2>(rational,
                                                         box.coefficients()));
}

TEST_CASE("Element values on a shell measure its area",
          "[element_values]")
{
    using Basis = TensorBSpline<double, 2>;

    const iguana::Patch<Basis, 3> shell = bent_shell();
    const iguana::GaussLegendre<double, 2> rule(4);
    iguana::ElementValues<Basis, 3> element_values(shell, {});

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    double area = 0.;

    for (const auto& element : shell.basis().grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, 2>::map_to_parameter_space(
            element.start(), element.end(), points, weights);
        element_values.reinit(element.index(), points);

        // The values keep the partition of unity at every point
        const Eigen::RowVectorXd sums =
            element_values.values().colwise().sum();
        REQUIRE((sums.array() - 1.).abs().maxCoeff() < 1e-12);

        area += weights.dot(element_values.measures());
    }

    REQUIRE_THAT(area, WithinAbs(sides[0] * sides[1], 1e-12));
}

TEST_CASE("Element values reject physical derivatives on a shell",
          "[element_values]")
{
    using ShellValues = iguana::ElementValues<TensorBSpline<double, 2>, 3>;

    const iguana::Patch<TensorBSpline<double, 2>, 3> shell = bent_shell();

    REQUIRE_THROWS_AS(ShellValues(shell, {.physical_gradients = true}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ShellValues(shell, {.physical_hessians = true}),
                      std::invalid_argument);
}
