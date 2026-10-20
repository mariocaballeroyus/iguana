/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/multi_index.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
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

/**
 * @brief Knot vector of one degree more with the continuities kept, each
 *        knot repeated once more, so that its splines hold those of the
 *        original
 */
KnotVector<double> elevated(const KnotVector<double>& knots)
{
    std::vector<double> values;

    for (const double knot : knots.values()) {
        if (values.empty() || knot != values.back())
            values.push_back(knot);

        values.push_back(knot);
    }

    return KnotVector<double>(knots.degree() + 1, std::move(values));
}

/// @brief Basis of one degree more on the elements of a basis, with its
///        continuities kept
template<std::size_t d>
TensorBSpline<double, d> elevated(const TensorBSpline<double, d>& basis)
{
    std::array<KnotVector<double>, d> knots = first_knot_vectors<d>();

    for (std::size_t direction = 0; direction < d; ++direction)
        knots[direction] = elevated(basis.grid().knots(direction));

    return TensorBSpline<double, d>{TensorGrid<double, d>(knots)};
}

/**
 * @brief Checks the functions of a basis of one degree more than the bent
 *        box, which hold the coordinates of its map: projected onto it, they
 *        have unit physical gradients and no physical second derivatives,
 *        with the map taken from the patch alone
 */
template<std::size_t d>
void check_elevated_basis()
{
    using Basis = TensorBSpline<double, d>;

    const iguana::Patch<Basis, d> patch = bent_box<d>();
    const Basis basis = elevated(patch.basis());
    const iguana::GaussLegendre<double, d> rule(4);

    // L2 projection of the coordinates onto the basis, in parameter space,
    // with the map from the patch's own functions at the same points
    iguana::ElementValues<Basis, d> functions(patch, basis, {});
    iguana::ElementValues<Basis, d> map(patch, {});
    Eigen::MatrixXd mass =
        Eigen::MatrixXd::Zero(basis.num_functions(), basis.num_functions());
    Eigen::MatrixXd moments = Eigen::MatrixXd::Zero(basis.num_functions(), d);

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    Eigen::VectorXi actives;
    Eigen::VectorXi map_actives;
    iguana::PointMatrix<double, d> positions;

    for (const auto& element : patch.basis().grid()) {
        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, d>::map_to_parameter_space(
            element.start(), element.end(), points, weights);
        functions.reinit(element.index(), points);
        map.reinit(element.index(), points);
        basis.active_on_element(element.index(), actives);
        patch.basis().active_on_element(element.index(), map_actives);
        patch.position_on_element(map_actives, map.values(), positions);

        const Eigen::MatrixXd& values = functions.values();
        mass(actives, actives) +=
            values * weights.asDiagonal() * values.transpose();
        moments(actives, Eigen::placeholders::all) +=
            values * weights.asDiagonal() * positions;
    }

    const Eigen::MatrixXd coordinates = mass.ldlt().solve(moments);

    iguana::ElementValues<Basis, d> element_values(
        patch, basis, {.physical_hessians = true});

    for (const auto& element : patch.basis().grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, d>::map_to_parameter_space(
            element.start(), element.end(), points, weights);
        element_values.reinit(element.index(), points);
        basis.active_on_element(element.index(), actives);

        const Eigen::MatrixXd u =
            coordinates(actives, Eigen::placeholders::all);

        for (std::size_t c = 0; c < d; ++c) {
            const Eigen::MatrixXd gradient =
                u.transpose() * element_values.physical_gradients()[c];
            Eigen::MatrixXd unit = Eigen::MatrixXd::Zero(d, points.rows());
            unit.row(static_cast<Eigen::Index>(c)).setOnes();

            REQUIRE((gradient - unit).cwiseAbs().maxCoeff() < 1e-10);
        }

        for (const Eigen::MatrixXd& hessians :
             element_values.physical_hessians())
            REQUIRE((u.transpose() * hessians).cwiseAbs().maxCoeff() < 1e-9);
    }
}

/// @brief Whether a matrix has the size of another and its entries, up to
///        rounding
bool agree(const Eigen::MatrixXd& computed, const Eigen::MatrixXd& expected)
{
    return computed.rows() == expected.rows()
           && computed.cols() == expected.cols()
           && computed.isApprox(expected, 1e-12);
}

/**
 * @brief Checks that the values of a basis of one degree more than the bent
 *        box and of the patch's own basis, held together, are those of each
 *        basis alone, after reinit(), shift() and differentiate()
 */
template<std::size_t d>
void check_several_bases()
{
    using Basis = TensorBSpline<double, d>;
    using Values = iguana::ElementValues<Basis, d>;

    const iguana::Patch<Basis, d> patch = bent_box<d>();
    const Basis basis = elevated(patch.basis());
    const iguana::ValueFlags flags{.physical_hessians = true};
    const iguana::GaussLegendre<double, d> rule(2);

    // The patch's own basis second, so that the map comes from a field
    // other than the first
    Values both(patch, {basis, patch.basis()}, flags);
    std::array<Values, 2> alone{Values(patch, basis, flags),
                                Values(patch, flags)};

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    for (const auto& element : patch.basis().grid()) {
        INFO("element " << element.index());

        rule.fill_to_reference_space(element.start(), element.end(), points,
                                     weights);
        iguana::BoxRule<double, d>::map_to_parameter_space(
            element.start(), element.end(), points, weights);

        both.reinit(element.index(), points);

        for (Values& values : alone)
            values.reinit(element.index(), points);

        REQUIRE(both.num_fields() == 2);
        REQUIRE(agree(both.measures(), alone[0].measures()));

        for (std::size_t field = 0; field < 2; ++field) {
            INFO("field " << field);
            const Values& single = alone[field];

            REQUIRE(agree(both.values(field), single.values()));

            for (std::size_t c = 0; c < d; ++c)
                REQUIRE(agree(both.physical_gradients(field)[c],
                              single.physical_gradients()[c]));

            for (std::size_t pair = 0; pair < d * (d + 1) / 2; ++pair)
                REQUIRE(agree(both.physical_hessians(field)[pair],
                              single.physical_hessians()[pair]));
        }

        // Each field expanded along the same shifts, small and different at
        // every point, then differentiated to its own degree across each
        // direction
        const Eigen::MatrixXd shifts = .01 * points.array().cos().matrix();
        both.shift(shifts, 2);

        for (Values& values : alone)
            values.shift(shifts, 2);

        for (std::size_t field = 0; field < 2; ++field)
            REQUIRE(agree(both.values(field), alone[field].values()));

        for (std::size_t direction = 0; direction < d; ++direction) {
            INFO("direction " << direction);
            both.differentiate(direction);

            for (Values& values : alone)
                values.differentiate(direction);

            for (std::size_t field = 0; field < 2; ++field)
                REQUIRE(agree(both.values(field), alone[field].values()));
        }
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

TEST_CASE("Element values take the map from the patch and the functions "
          "from a basis of higher degree", "[element_values]")
{
    check_elevated_basis<2>();
    check_elevated_basis<3>();
}

TEST_CASE("Element values of several bases are those of each basis alone",
          "[element_values]")
{
    check_several_bases<2>();
    check_several_bases<3>();
}

TEST_CASE("Element values need bases, all on the elements of the patch",
          "[element_values]")
{
    using Basis = TensorBSpline<double, 2>;
    using Bases = std::vector<std::reference_wrapper<const Basis>>;

    const iguana::Patch<Basis, 2> patch = bent_box<2>();
    const Basis moved{TensorGrid<double, 2>(
        {KnotVector<double>(2, {0., 0., 0., .3, .6, 1., 1., 1.}),
         knot_vectors[1]})};

    REQUIRE_THROWS_AS((iguana::ElementValues<Basis, 2>(patch, moved, {})),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((iguana::ElementValues<Basis, 2>(patch, Bases{}, {})),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((iguana::ElementValues<Basis, 2>(
                          patch, Bases{patch.basis(), moved}, {})),
                      std::invalid_argument);
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
