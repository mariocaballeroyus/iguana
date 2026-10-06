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
 */
template<std::size_t d>
iguana::Patch<TensorBSpline<double, d>, d> bent_box()
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
                    .03 * sides[direction]
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

} // namespace

TEST_CASE("Element values integrate the energy of a linear field exactly",
          "[element_values]")
{
    check_linear_field<2>();
    check_linear_field<3>();
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

TEST_CASE("Element values reject physical gradients on a shell",
          "[element_values]")
{
    using ShellValues = iguana::ElementValues<TensorBSpline<double, 2>, 3>;

    const iguana::Patch<TensorBSpline<double, 2>, 3> shell = bent_shell();
    const iguana::ValueFlags flags{.physical_gradients = true};

    REQUIRE_THROWS_AS(ShellValues(shell, flags), std::invalid_argument);
}
