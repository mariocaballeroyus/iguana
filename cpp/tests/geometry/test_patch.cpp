/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;
using iguana::Patch;
using iguana::TensorBSpline;
using iguana::TensorNURBS;

using Points = iguana::PointMatrix<double, 3>;

BSpline<double> quadratic()
{
    return BSpline<double>(2, {0., 0., 0., .4, 1., 1., 1.});
}

BSpline<double> cubic()
{
    return BSpline<double>(3, {0., 0., 0., 0., .6, 1., 1., 1., 1.});
}

BSpline<double> repeated()
{
    return BSpline<double>(2, {0., 0., 0., .5, .5, 1., 1., 1.});
}

double greville(const BSpline<double>& axis, int function)
{
    double sum = 0.;

    for (int knot = 1; knot <= axis.degree(); ++knot)
        sum += axis.knots().values()[function + knot];

    return sum / axis.degree();
}

/**
 * @brief Control points of an affine map, sampled at the Greville abscissae,
 *        which reproduce it exactly
 *
 * @param basis Basis of the patch
 * @param map Linear part of the map, from parameters to physical space
 * @param shift Constant part of the map
 */
template<std::size_t d, std::size_t n>
iguana::PointMatrix<double, n> affine_coefficients(
    const TensorBSpline<double, d>& basis,
    const Eigen::Matrix<double, n, d>& map,
    const Eigen::Vector<double, n>& shift)
{
    iguana::PointMatrix<double, n> coefficients(basis.num_functions(), n);
    Eigen::Vector<double, d> node;

    for (int fun = 0; fun < basis.num_functions(); ++fun) {
        int rest = fun;

        for (std::size_t dir = 0; dir < d; ++dir) {
            const BSpline<double>& axis = basis.axis(dir);

            node[static_cast<Eigen::Index>(dir)] =
                greville(axis, rest % axis.num_functions());
            rest /= axis.num_functions();
        }

        coefficients.row(fun) = (map * node + shift).transpose();
    }

    return coefficients;
}

/**
 * @brief Checks that a patch reproduces an affine map and its tangents at
 *        points of every element, knows that its map is affine and, with as
 *        many directions as dimensions, inverts the points back
 *
 * @param basis Basis of the patch
 * @param map Linear part of the map, from parameters to physical space
 * @param shift Constant part of the map
 */
template<std::size_t d, std::size_t n>
void check_affine(const TensorBSpline<double, d>& basis,
                  const Eigen::Matrix<double, n, d>& map,
                  const Eigen::Vector<double, n>& shift)
{
    INFO("directions " << d << ", dimensions " << n);

    const Patch<TensorBSpline<double, d>, n> patch(
        basis, affine_coefficients<d, n>(basis, map, shift));

    REQUIRE(patch.is_affine());

    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    iguana::PointMatrix<double, n> positions;
    std::array<Eigen::MatrixXd, d> gradients;
    std::array<iguana::PointMatrix<double, n>, d> tangents;
    Eigen::VectorXd measures;

    for (int element = 0; element < basis.grid().num_elements();
         ++element) {
        INFO("element " << element);
        std::array<int, d> first{};
        Eigen::MatrixXd points(2, d);
        int rest = element;

        for (std::size_t dir = 0; dir < d; ++dir) {
            const BSpline<double>& axis = basis.axis(dir);
            const iguana::KnotVector<double>& knots = axis.knots();
            const int index = rest % knots.num_elements();
            const double start = knots.element_start(index);
            const double width = knots.element_end(index) - start;
            const double shear = .1 * static_cast<double>(dir);

            rest /= knots.num_elements();
            first[dir] = axis.first_active(index);

            // Shear the fractions so that swapping a point index for a
            // coordinate index would move the points
            points(0, dir) = start + (.25 + shear) * width;
            points(1, dir) = start + (.75 - shear) * width;
        }

        basis.active_on_element(element, actives);
        basis.eval_on_element(first, points, values);
        patch.position_on_element(actives, values, positions);

        REQUIRE(positions.rows() == points.rows());

        for (Eigen::Index pt = 0; pt < points.rows(); ++pt) {
            INFO("point " << pt);
            const Eigen::Vector<double, n> exact =
                map * points.row(pt).transpose() + shift;

            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(n); ++c)
                REQUIRE_THAT(positions(pt, c), WithinAbs(exact[c], 1e-13));
        }

        // A region or a volume maps its positions back to their parameters
        if constexpr (d == n) {
            iguana::PointMatrix<double, d> parameters;
            patch.invert_points(positions, parameters);

            for (Eigen::Index pt = 0; pt < points.rows(); ++pt)
                for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(d); ++c)
                    REQUIRE_THAT(parameters(pt, c),
                                 WithinAbs(points(pt, c), 1e-13));
        }

        // The tangents of an affine map are the columns of its linear part
        basis.grad_on_element(first, points, values, gradients);
        patch.tangent_on_element(actives, gradients, tangents);

        for (std::size_t dir = 0; dir < d; ++dir) {
            INFO("direction " << dir);
            const Eigen::Index column = static_cast<Eigen::Index>(dir);

            for (Eigen::Index pt = 0; pt < points.rows(); ++pt)
                for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(n); ++c)
                    REQUIRE_THAT(tangents[dir](pt, c),
                                 WithinAbs(map(c, column), 1e-12));
        }

        // Its measure is that of its linear part, the same at every point
        double measure = 0.;

        if constexpr (d == n)
            measure = std::abs(map.determinant());
        else if constexpr (d < n)
            measure = std::sqrt((map.transpose() * map).determinant());

        Patch<TensorBSpline<double, d>, n>::measure_on_element(tangents,
                                                               measures);

        for (Eigen::Index pt = 0; pt < points.rows(); ++pt)
            REQUIRE_THAT(measures(pt), WithinAbs(measure, 1e-12));
    }
}

/// @brief Half annulus of radii 1 and 2 around the origin, the angle along
///        the first direction and the radius along the second
Patch<TensorNURBS<double, 2>, 2> half_annulus()
{
    // Two quarter circles of unit radius joined at a double knot, swept
    // along a radius running linearly from 1 to 2
    const TensorBSpline<double, 2> bspline(
        {BSpline<double>(2, {0., 0., 0., .5, .5, 1., 1., 1.}),
         BSpline<double>(1, {0., 0., 1., 1.})});

    Eigen::Matrix<double, 5, 2> circle;
    circle << 1., 0.,
              1., 1.,
              0., 1.,
             -1., 1.,
             -1., 0.;

    const double corner = std::sqrt(.5);
    Eigen::Vector<double, 5> circle_weights;
    circle_weights << 1., corner, 1., corner, 1.;

    // The weights vary along the first direction alone, so that a weight
    // gathered along the wrong direction breaks the circle
    Eigen::VectorXd weights(bspline.num_functions());
    iguana::PointMatrix<double, 2> coefficients(bspline.num_functions(), 2);

    for (int ring = 0; ring < 2; ++ring) {
        for (int arc = 0; arc < 5; ++arc) {
            weights(arc + 5 * ring) = circle_weights(arc);
            coefficients.row(arc + 5 * ring) = (1. + ring) * circle.row(arc);
        }
    }

    return {TensorNURBS<double, 2>(bspline, weights), coefficients};
}

} // namespace

TEST_CASE("Patch requires one control point per basis function", "[patch]")
{
    using Surface = Patch<TensorBSpline<double, 2>, 3>;

    const TensorBSpline<double, 2> basis({quadratic(), cubic()});
    const int expected = basis.num_functions();

    REQUIRE_THROWS_AS(Surface(basis,
                              Points::Zero(expected - 1, 3)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Surface(basis,
                              Points::Zero(expected + 1, 3)),
                      std::invalid_argument);
    REQUIRE_NOTHROW(Surface(basis, Points::Zero(expected, 3)));
}

TEST_CASE("Patch reproduces affine maps exactly", "[patch]")
{
    Eigen::Matrix<double, 3, 1> curve;
    curve << 2., -1., .5;

    Eigen::Matrix<double, 3, 2> surface;
    surface << 2., .5,
              -1., 3.,
               .25, -2.;

    Eigen::Matrix<double, 3, 3> volume;
    volume << 2., .5, -1.,
             -1., 3., .75,
              .25, -2., 1.5;

    const Eigen::Vector3d shift(1.5, -.25, 3.);

    // The zero map leaves the constant map, which holds by partition of
    // unity alone
    check_affine<2, 3>(TensorBSpline<double, 2>({quadratic(), cubic()}),
                       Eigen::Matrix<double, 3, 2>::Zero(), shift);

    // Curves, surfaces and volumes in space
    check_affine<1, 3>(TensorBSpline<double, 1>({cubic()}), curve, shift);
    check_affine<2, 3>(TensorBSpline<double, 2>({quadratic(), cubic()}),
                       surface, shift);
    check_affine<3, 3>(TensorBSpline<double, 3>({quadratic(), cubic(),
                                                 repeated()}),
                       volume, shift);

    // Curves and regions in the plane
    check_affine<1, 2>(TensorBSpline<double, 1>({cubic()}),
                       curve.topRows<2>(), shift.head<2>());
    check_affine<2, 2>(TensorBSpline<double, 2>({quadratic(), cubic()}),
                       surface.topRows<2>(), shift.head<2>());
}

TEST_CASE("Patch tells maps that are not affine and does not invert them",
          "[patch]")
{
    using Region = Patch<TensorBSpline<double, 2>, 2>;
    using NURBSRegion = Patch<TensorNURBS<double, 2>, 2>;

    Eigen::Matrix2d map;
    map << 2., .5,
          -1., 3.;

    const TensorBSpline<double, 2> basis({quadratic(), cubic()});
    const iguana::PointMatrix<double, 2> affine =
        affine_coefficients<2, 2>(basis, map, Eigen::Vector2d(1.5, -.25));

    // Equal weights leave the functions B-splines, unequal ones do not
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(basis.num_functions(),
                                                        2.);
    REQUIRE(NURBSRegion(TensorNURBS<double, 2>(basis, weights), affine)
                .is_affine());

    weights(1) = 3.;
    REQUIRE_FALSE(NURBSRegion(TensorNURBS<double, 2>(basis, weights), affine)
                      .is_affine());

    // Constant pieces do not reproduce the identity
    const TensorBSpline<double, 2> pieces(
        {BSpline<double>(0, {0., .5, 1.}), cubic()});
    iguana::PointMatrix<double, 2> steps(pieces.num_functions(), 2);

    for (int function = 0; function < pieces.num_functions(); ++function)
        steps.row(function) << function, 1.;

    REQUIRE_FALSE(Region(pieces, steps).is_affine());

    // Moving one control point off the affine map bends it
    iguana::PointMatrix<double, 2> moved = affine;
    moved(5, 0) += 1e-3;

    const Region bent(basis, moved);
    iguana::PointMatrix<double, 2> parameters;

    REQUIRE_FALSE(bent.is_affine());
    REQUIRE_THROWS_AS(bent.invert_points(moved, parameters),
                      std::invalid_argument);
}

TEST_CASE("NURBS patch reproduces a half annulus exactly", "[patch]")
{
    const Patch<TensorNURBS<double, 2>, 2> patch = half_annulus();
    const TensorNURBS<double, 2>& basis = patch.basis();

    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    iguana::PointMatrix<double, 2> positions;
    std::array<Eigen::MatrixXd, 2> gradients;
    std::array<iguana::PointMatrix<double, 2>, 2> tangents;

    for (const auto& element : basis.grid()) {
        INFO("element " << element.index());
        Eigen::MatrixXd points(2, 2);

        for (std::size_t dir = 0; dir < 2; ++dir) {
            const double start = element.start()[dir];
            const double width = element.end()[dir] - start;
            const double shear = .1 * static_cast<double>(dir);

            points(0, dir) = start + (.25 + shear) * width;
            points(1, dir) = start + (.75 - shear) * width;
        }

        basis.active_on_element(element.index(), actives);
        basis.grad_on_element(element.first_active(), points, values,
                              gradients);
        patch.position_on_element(actives, values, positions);
        patch.tangent_on_element(actives, gradients, tangents);

        for (Eigen::Index pt = 0; pt < points.rows(); ++pt) {
            INFO("point " << pt);
            const Eigen::Vector2d position = positions.row(pt).transpose();
            const double radius = position.norm();

            // The radius follows the second parameter, so the first moves
            // along a circle and the second along a ray
            REQUIRE_THAT(radius, WithinAbs(1. + points(pt, 1), 1e-14));
            REQUIRE_THAT(position.dot(tangents[0].row(pt).transpose()),
                         WithinAbs(0., 1e-13));

            for (Eigen::Index c = 0; c < 2; ++c)
                REQUIRE_THAT(tangents[1](pt, c),
                             WithinAbs(position[c] / radius, 1e-13));
        }
    }
}

TEST_CASE("NURBS patch measures its area and maps gradients to physical "
          "ones", "[patch]")
{
    const Patch<TensorNURBS<double, 2>, 2> patch = half_annulus();
    const TensorNURBS<double, 2>& basis = patch.basis();
    const iguana::GaussLegendre<double, 2> rule({8, 8});

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    Eigen::VectorXd measures;
    std::array<Eigen::MatrixXd, 2> gradients;
    std::array<Eigen::MatrixXd, 2> physical_gradients;
    std::array<iguana::PointMatrix<double, 2>, 2> tangents;
    double area = 0.;

    for (const auto& element : basis.grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, 2>::map_to_parameter_space(
            element.start(), element.end(), points, weights);

        basis.active_on_element(element.index(), actives);
        basis.grad_on_element(element.first_active(), points, values,
                              gradients);
        patch.tangent_on_element(actives, gradients, tangents);
        Patch<TensorNURBS<double, 2>, 2>::measure_on_element(tangents,
                                                             measures);
        Patch<TensorNURBS<double, 2>, 2>::physical_grad_on_element(
            tangents, gradients, physical_gradients);

        area += weights.dot(measures);

        // The map is the sum of the control points times their functions,
        // so its physical gradient, the identity, is that of the functions
        const iguana::PointMatrix<double, 2> net =
            patch.coefficients()(actives, Eigen::placeholders::all);

        for (Eigen::Index pt = 0; pt < points.rows(); ++pt) {
            INFO("point " << pt);
            Eigen::Matrix2d identity;

            for (Eigen::Index c = 0; c < 2; ++c)
                identity.col(c) =
                    net.transpose() * physical_gradients[c].col(pt);

            REQUIRE((identity - Eigen::Matrix2d::Identity())
                        .cwiseAbs().maxCoeff() < 1e-13);
        }
    }

    // A rational map is integrated to the convergence of the Gauss rule
    REQUIRE_THAT(area, WithinAbs(1.5 * std::numbers::pi, 1e-9));
}

TEST_CASE("Patch measures a boundary and maps its normals", "[patch]")
{
    using Region = Patch<TensorNURBS<double, 2>, 2>;
    using Surface = Patch<TensorNURBS<double, 2>, 3>;

    const Region region = half_annulus();
    const TensorNURBS<double, 2>& basis = region.basis();

    // The same half annulus lifted into space by a rotation, a shell
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(.7, Eigen::Vector3d(1., 2., 3.).normalized())
            .toRotationMatrix();
    Points lifted = Points::Zero(basis.num_functions(), 3);
    lifted.leftCols<2>() = region.coefficients();
    const Points rotated = lifted * rotation.transpose();
    const Surface surface(basis, rotated);

    // Points of the second element, and the normals of a circle and of a
    // ray through them
    Eigen::MatrixXd points(2, 2);
    points << .6, .3,
              .85, .7;
    const Eigen::MatrixXd circle = Eigen::RowVector2d(0., 1.).replicate(2, 1);
    const Eigen::MatrixXd ray = Eigen::RowVector2d(1., 0.).replicate(2, 1);
    const Eigen::MatrixXd oblique =
        Eigen::RowVector2d(.6, .8).replicate(2, 1);

    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    std::array<Eigen::MatrixXd, 2> gradients;
    iguana::PointMatrix<double, 2> positions;
    std::array<iguana::PointMatrix<double, 2>, 2> tangents;
    std::array<Points, 2> surface_tangents;

    basis.active_on_element(1, actives);
    basis.grad_on_element({2, 0}, points, values, gradients);
    region.position_on_element(actives, values, positions);
    region.tangent_on_element(actives, gradients, tangents);
    surface.tangent_on_element(actives, gradients, surface_tangents);

    Eigen::VectorXd circles;
    Eigen::VectorXd rays;
    Eigen::VectorXd surface_circles;

    iguana::PointMatrix<double, 2> circle_normals;
    iguana::PointMatrix<double, 2> ray_normals;
    Points surface_circle_normals;

    Region::boundary_measure_on_element(tangents, circle, circles);
    Region::boundary_measure_on_element(tangents, ray, rays);
    Surface::boundary_measure_on_element(surface_tangents, circle,
                                         surface_circles);

    Eigen::VectorXd obliques;
    iguana::PointMatrix<double, 2> oblique_normals;

    Region::boundary_measure_on_element(tangents, oblique, obliques);
    Region::physical_normal_on_element(tangents, oblique, oblique_normals);
    Region::physical_normal_on_element(tangents, circle, circle_normals);
    Region::physical_normal_on_element(tangents, ray, ray_normals);
    Surface::physical_normal_on_element(surface_tangents, circle,
                                        surface_circle_normals);

    for (Eigen::Index pt = 0; pt < points.rows(); ++pt) {
        INFO("point " << pt);
        const Eigen::Vector2d radial = positions.row(pt).normalized();
        const Eigen::Vector2d angular(-radial.y(), radial.x());

        // A circle has the radial normal and is measured by the speed of the
        // angle along the first direction, and a ray has the angular normal
        // and the unit speed of the radius
        REQUIRE((circle_normals.row(pt).transpose() - radial).norm() < 1e-13);
        REQUIRE_THAT(circles(pt),
                     WithinAbs(tangents[0].row(pt).norm(), 1e-13));
        REQUIRE((ray_normals.row(pt).transpose() - angular).norm() < 1e-13);
        REQUIRE_THAT(rays(pt), WithinAbs(1., 1e-13));

        // An oblique line runs along J t, with t the normal turned a quarter
        // in parameter space: its normal is orthogonal to J t, on the side of
        // J m, and its measure is |J t|
        Eigen::Matrix2d jacobian;
        jacobian << tangents[0].row(pt).transpose(),
                    tangents[1].row(pt).transpose();
        const Eigen::Vector2d along = jacobian * Eigen::Vector2d(-.8, .6);
        const Eigen::Vector2d across = jacobian * Eigen::Vector2d(.6, .8);
        const Eigen::Vector2d normal = oblique_normals.row(pt).transpose();

        REQUIRE_THAT(normal.dot(along), WithinAbs(0., 1e-13));
        REQUIRE(normal.dot(across) > 0.);
        REQUIRE_THAT(obliques(pt), WithinAbs(along.norm(), 1e-13));

        // On the shell, the co-normal is the planar normal rotated, tangent
        // to the surface, and the boundary curve measures the same
        Eigen::Vector3d lifted_radial = Eigen::Vector3d::Zero();
        lifted_radial.head<2>() = radial;

        REQUIRE((surface_circle_normals.row(pt).transpose()
                 - rotation * lifted_radial).norm() < 1e-13);
        REQUIRE_THAT(surface_circles(pt), WithinAbs(circles(pt), 1e-13));
    }
}
