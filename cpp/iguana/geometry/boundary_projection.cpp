/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary_projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/field_interpolation.hpp"
#include "iguana/geometry/nurbs/bezier_points.hpp"
#include "iguana/geometry/nurbs/curve_projection.hpp"
#include "iguana/grid/knot_vector.hpp"
#include "iguana/quadrature/gauss_legendre/gauss_legendre.hpp"
#include "iguana/utils/taylor_series.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Series of x' and y' in the parameter about a point of a face, with
 *        coefficients x^(k + 1) / k! for k below a count
 *
 * @pre @p parameter lies in @p element of the curve
 */
template<std::floating_point T>
void tangent_series(const Patch<TensorNURBS<T, 1>, 2>& curve, int element,
                    T parameter, int count, Eigen::VectorX<T>& dx,
                    Eigen::VectorX<T>& dy)
{
    const BSpline<T>& bspline = curve.basis().bspline().axis(0);
    Eigen::MatrixX<T> point(1, 1);
    std::vector<Eigen::MatrixX<T>> derivs;
    Eigen::VectorXi actives;
    PointMatrix<T, 2> derivative;

    point(0, 0) = parameter;
    curve.basis().derivs_on_element({bspline.first_active(element)}, point,
                                    count, derivs);
    curve.basis().active_on_element(element, actives);

    dx.resize(count);
    dy.resize(count);
    T factorial{1};

    for (int k = 0; k < count; ++k) {
        if (k > 0)
            factorial *= static_cast<T>(k);

        curve.position_on_element(
            actives, derivs[static_cast<std::size_t>(k) + 1], derivative);
        dx(k) = derivative(0, 0) / factorial;
        dy(k) = derivative(0, 1) / factorial;
    }
}

/**
 * @brief Derivatives along the arc length of a parametrization at the point
 *        a series in its parameter is taken about, d / ds = |c'|^(−1) d / du
 *
 * @param series Taylor series of the quantity in the parameter, with at
 *        least order + 1 coefficients
 * @param inverse_speed Taylor series of |c'|^(−1), with at least order
 *        coefficients
 * @param order Highest order of the derivatives
 * @return Derivatives of order 0 to @p order
 */
template<std::floating_point T>
Eigen::VectorX<T> arc_length_derivs(Eigen::VectorX<T> series,
                                    const Eigen::VectorX<T>& inverse_speed,
                                    int order)
{
    Eigen::VectorX<T> derivs(order + 1);

    for (int j = 0; j <= order; ++j) {
        derivs(j) = series(0);

        if (j < order)
            series = series_product(inverse_speed, series_derivative(series));
    }

    return derivs;
}

/// @brief Length of an element of a face, by Gauss quadrature of the speed
///        of its curve
template<std::floating_point T>
T element_length(const Patch<TensorNURBS<T, 1>, 2>& curve, int element)
{
    const BSpline<T>& bspline = curve.basis().bspline().axis(0);
    const KnotVector<T>& knots = bspline.knots();
    const GaussLegendre<T, 1> rule(2 * (knots.degree() + 1));
    const std::array<T, 1> start{knots.element_start(element)};
    const std::array<T, 1> end{knots.element_end(element)};
    Eigen::MatrixX<T> points;
    Eigen::VectorX<T> weights;
    std::vector<Eigen::MatrixX<T>> derivs;
    Eigen::VectorXi actives;
    PointMatrix<T, 2> tangents;

    rule.fill_to_reference_space(start, end, points, weights);
    GaussLegendre<T, 1>::map_to_parameter_space(start, end, points, weights);
    curve.basis().derivs_on_element({bspline.first_active(element)}, points,
                                    1, derivs);
    curve.basis().active_on_element(element, actives);
    curve.position_on_element(actives, derivs[1], tangents);

    return weights.dot(tangents.rowwise().norm());
}

/**
 * @brief Knot vector of the data spline of a face, clamped, of a degree,
 *        with each element of the curve split evenly into as many as it is
 *        long in a size, at least one
 *
 * @details The data along a curve are no smoother in its parameter than
 *          the curve is at its knots, C^(p − r) for a knot of multiplicity r
 *          in a curve of degree p, so each knot of the curve is one of the
 *          data spline with the multiplicity that leaves it that smooth
 */
template<std::floating_point T>
KnotVector<T> data_knots(const Patch<TensorNURBS<T, 1>, 2>& curve,
                         int degree, T size)
{
    const KnotVector<T>& knots = curve.basis().bspline().axis(0).knots();
    const int last = knots.num_elements() - 1;
    std::vector<T> values(static_cast<std::size_t>(degree) + 1,
                          knots.domain_start());

    for (int element = 0; element <= last; ++element) {
        const T start = knots.element_start(element);
        const T end = knots.element_end(element);
        const int count = std::max(1, static_cast<int>(std::ceil(
                                          element_length(curve, element)
                                          / size)));

        for (int knot = 1; knot < count; ++knot)
            values.push_back(start + (end - start) * static_cast<T>(knot)
                                         / static_cast<T>(count));

        if (element == last)
            break;

        const int repeats = static_cast<int>(
            std::count(knots.values().begin(), knots.values().end(), end));
        const int continuity = knots.degree() - repeats;
        const int multiplicity = std::clamp(degree - continuity, 1, degree);

        values.insert(values.end(), static_cast<std::size_t>(multiplicity),
                      end);
    }

    values.insert(values.end(), static_cast<std::size_t>(degree) + 1,
                  knots.domain_end());

    return KnotVector<T>(degree, std::move(values));
}

/// @brief Smallest edge of the physical cells of a patch with an affine
///        map, the shortest element of each direction through its column
///        of the map
template<typename Basis>
typename Basis::Scalar smallest_cell_edge(const Patch<Basis, 2>& patch)
{
    using T = typename Basis::Scalar;

    T smallest = std::numeric_limits<T>::infinity();

    for (std::size_t direction = 0; direction < 2; ++direction) {
        const KnotVector<T>& knots =
            patch.basis().grid().knots(direction);
        const T scale = patch.linear_part()
                            .col(static_cast<Eigen::Index>(direction))
                            .norm();

        for (int element = 0; element < knots.num_elements(); ++element)
            smallest = std::min(smallest, scale * (knots.element_end(element)
                                                   - knots.element_start(
                                                       element)));
    }

    return smallest;
}

/**
 * @brief Closest points of a boundary of curves to positions in the plane,
 *        the search behind project_closest_points()
 *
 * @throws std::invalid_argument As project_closest_points() does for the
 *         boundary
 */
template<std::floating_point T>
BoundaryProjection<T> project_closest_phys(const Boundary<T, 2>& boundary,
                                           const PointMatrix<T, 2>& positions)
{
    using Face = typename Boundary<T, 2>::Face;

    if (boundary.num_faces() == 0)
        throw std::invalid_argument("project_closest_points: "
                                    "there must be a face to project onto");

    // Width below which sign changes pair up, as for the crossings
    constexpr T tolerance = 4096 * std::numeric_limits<T>::epsilon();

    // Homogeneous Bezier points of every element of every face, which throw
    // unless the knot vectors are clamped
    struct FaceElement
    {
        int face;
        int element;
        Eigen::MatrixX<T> points;
    };

    std::vector<FaceElement> searched;

    for (int face = 0; face < boundary.num_faces(); ++face) {
        const Face& curve = boundary.face(face);
        const std::vector<Eigen::MatrixX<T>> operators =
            extraction_operators(curve.basis().bspline().axis(0).knots());

        for (int element = 0; element < static_cast<int>(operators.size());
             ++element) {
            const Eigen::MatrixX<T>& extraction =
                operators[static_cast<std::size_t>(element)];
            searched.push_back(
                {face, element, bezier_points(curve, element, extraction)});
        }
    }

    const Eigen::Index num_points = positions.rows();

    BoundaryProjection<T> projection;
    projection.faces.resize(num_points);
    projection.elements.resize(num_points);
    projection.parameters.resize(num_points);
    projection.positions.resize(num_points, 2);
    projection.normals.resize(num_points, 2);

    // Buffers reused over the points
    Eigen::MatrixX<T> parameter(1, 1);
    Eigen::MatrixX<T> values;
    std::array<Eigen::MatrixX<T>, 1> gradients;
    Eigen::VectorXi actives;
    PointMatrix<T, 2> position;
    PointMatrix<T, 2> normal;

    for (Eigen::Index point = 0; point < num_points; ++point) {
        const Eigen::RowVectorX<T> target = positions.row(point);

        // Nearest element, the first of those at one distance
        const FaceElement* nearest = nullptr;
        T closest_t{0};
        T closest_distance = std::numeric_limits<T>::infinity();

        for (const FaceElement& candidate : searched) {
            const auto [t, distance] =
                closest_on_element(candidate.points, target, tolerance);

            if (distance < closest_distance) {
                nearest = &candidate;
                closest_t = t;
                closest_distance = distance;
            }
        }

        // Parameter of the face
        const Face& curve = boundary.face(nearest->face);
        const BSpline<T>& bspline = curve.basis().bspline().axis(0);
        const T start = bspline.knots().element_start(nearest->element);
        const T end = bspline.knots().element_end(nearest->element);

        parameter(0, 0) =
            closest_t == T{1} ? end : start + closest_t * (end - start);

        // The closest point and its normal, through the face
        const std::array<int, 1> first_active{
            bspline.first_active(nearest->element)};

        curve.basis().grad_on_element(first_active, parameter, values,
                                      gradients);
        curve.basis().active_on_element(nearest->element, actives);
        curve.position_on_element(actives, values, position);
        boundary.normal_on_element(nearest->face, actives, gradients, normal);

        projection.faces(point) = nearest->face;
        projection.elements(point) = nearest->element;
        projection.parameters(point) = parameter(0, 0);
        projection.positions.row(point) = position.row(0);
        projection.normals.row(point) = normal.row(0);
    }

    return projection;
}

} // namespace

template<typename Basis> requires (Basis::dimension == 2)
BoundaryProjection<typename Basis::Scalar> project_closest_points(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary,
    const PointMatrix<typename Basis::Scalar, 2>& points)
{
    using T = typename Basis::Scalar;

    if (!patch.is_affine())
        throw std::invalid_argument("project_closest_points: "
                                    "the map of the patch must be affine");

    // Use the patch for the physical points, through the map, x = a + A xi
    PointMatrix<T, 2> positions = points * patch.linear_part().transpose();
    positions.rowwise() += patch.offset().transpose();

    // Closest points on the boundary in physical space
    BoundaryProjection<T> projection =
        project_closest_phys(boundary, positions);

    // Pulled back into parameter space
    PointMatrix<T, 2> closest;
    patch.invert_points(projection.positions, closest);

    projection.distances = closest - points;

    return projection;
}

template<std::floating_point T>
Eigen::MatrixX<T> curvature_derivs(const Boundary<T, 2>& boundary,
                                   const BoundaryProjection<T>& projection,
                                   int order)
{
    using Face = typename Boundary<T, 2>::Face;

    if (order < 0)
        throw std::invalid_argument("curvature_derivs: "
                                    "the order must be non-negative");

    const Eigen::Index num_points = projection.faces.size();
    Eigen::MatrixX<T> curvatures(num_points, order + 1);
    Eigen::VectorX<T> dx;
    Eigen::VectorX<T> dy;

    for (Eigen::Index point = 0; point < num_points; ++point) {
        const int face = projection.faces(point);
        const Face& curve = boundary.face(face);

        tangent_series(curve, projection.elements(point),
                       projection.parameters(point), order + 2, dx, dy);

        // κ = (x' y'' − y' x'') / |c'|^3 along the parametrization
        const Eigen::VectorX<T> ddx = series_derivative(dx);
        const Eigen::VectorX<T> ddy = series_derivative(dy);
        const Eigen::VectorX<T> cross =
            series_product(dx, ddy) - series_product(dy, ddx);
        const Eigen::VectorX<T> speed_squared =
            series_product(dx, dx) + series_product(dy, dy);
        const Eigen::VectorX<T> derivs = arc_length_derivs(
            series_product(cross, series_power(speed_squared, T{-1.5})),
            series_power(speed_squared, T{-0.5}), order);

        // τ follows the parametrization where the face sign is 1 and runs
        // against it otherwise, with ν and κ turning with the sign
        const T sign = static_cast<T>(boundary.sign(face));
        T factor = sign;

        for (int j = 0; j <= order; ++j) {
            curvatures(point, j) = factor * derivs(j);
            factor *= sign;
        }
    }

    return curvatures;
}

template<typename Basis> requires (Basis::dimension == 2)
PointMatrix<typename Basis::Scalar, 2> boundary_interpolation_points(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary, int degree)
{
    using T = typename Basis::Scalar;
    using Face = typename Boundary<T, 2>::Face;

    if (degree < 0)
        throw std::invalid_argument("boundary_interpolation_points: "
                                    "the degree must be non-negative");

    if (!patch.is_affine())
        throw std::invalid_argument("boundary_interpolation_points: "
                                    "the map of the patch must be affine");

    const T size = smallest_cell_edge(patch);
    std::vector<Eigen::RowVector2<T>> points;
    Eigen::MatrixX<T> parameter(1, 1);
    std::vector<Eigen::MatrixX<T>> values;
    Eigen::VectorXi actives;
    PointMatrix<T, 2> position;

    for (int face = 0; face < boundary.num_faces(); ++face) {
        const Face& curve = boundary.face(face);
        const BSpline<T>& bspline = curve.basis().bspline().axis(0);
        const Eigen::VectorX<T> abscissae =
            greville_abscissae(data_knots(curve, degree, size));

        // The curve at each abscissa, on the element of its own knot
        // vector holding it
        for (const T abscissa : abscissae) {
            const int element = element_holding(bspline.knots(), abscissa);

            parameter(0, 0) = abscissa;
            curve.basis().derivs_on_element({bspline.first_active(element)},
                                            parameter, 0, values);
            curve.basis().active_on_element(element, actives);
            curve.position_on_element(actives, values[0], position);
            points.push_back(position.row(0));
        }
    }

    PointMatrix<T, 2> result(static_cast<Eigen::Index>(points.size()), 2);

    for (std::size_t point = 0; point < points.size(); ++point)
        result.row(static_cast<Eigen::Index>(point)) = points[point];

    return result;
}

template<typename Basis> requires (Basis::dimension == 2)
Eigen::MatrixX<typename Basis::Scalar> boundary_interpolated_derivs(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary, int degree,
    const Eigen::VectorX<typename Basis::Scalar>& values,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order)
{
    using T = typename Basis::Scalar;
    using Face = typename Boundary<T, 2>::Face;

    if (order < 0 || order > degree)
        throw std::invalid_argument("boundary_interpolated_derivs: "
                                    "the order must lie between zero and "
                                    "the degree");

    if (!patch.is_affine())
        throw std::invalid_argument("boundary_interpolated_derivs: "
                                    "the map of the patch must be affine");

    // The data spline of each face and its coefficients, from the values at
    // its Greville abscissae, face after face
    const T size = smallest_cell_edge(patch);
    std::vector<BSpline<T>> splines;
    std::vector<Eigen::VectorX<T>> coefficients;
    Eigen::Index first = 0;

    for (int face = 0; face < boundary.num_faces(); ++face) {
        splines.emplace_back(data_knots(boundary.face(face), degree, size));
        const Eigen::Index count = splines.back().num_functions();

        if (first + count > values.size())
            break;

        coefficients.push_back(collocation_matrix(splines.back())
                                   .partialPivLu()
                                   .solve(values.segment(first, count)));
        first += count;
    }

    if (first != values.size()
        || static_cast<int>(coefficients.size()) != boundary.num_faces())
        throw std::invalid_argument("boundary_interpolated_derivs: "
                                    "the values must be one per "
                                    "interpolation point");

    const Eigen::Index num_points = projection.faces.size();
    Eigen::MatrixX<T> derivs(num_points, order + 1);
    std::vector<Eigen::MatrixX<T>> spline_derivs;
    Eigen::VectorX<T> dx;
    Eigen::VectorX<T> dy;
    Eigen::VectorX<T> series(order + 1);

    for (Eigen::Index point = 0; point < num_points; ++point) {
        const int face = projection.faces(point);
        const Face& curve = boundary.face(face);
        const BSpline<T>& spline = splines[static_cast<std::size_t>(face)];
        const T parameter = projection.parameters(point);

        // Series of g_h in the parameter about the closest point, with
        // coefficients g^(k) / k!, on the data element holding it inside
        // its element of the curve, which the knots of the data split, so
        // that at a knot it is one-sided as the curvature is
        const KnotVector<T>& curve_knots =
            curve.basis().bspline().axis(0).knots();
        const T curve_end =
            curve_knots.element_end(projection.elements(point));
        int element = element_holding(spline.knots(), parameter);

        if (element > 0 && spline.knots().element_start(element) >= curve_end)
            --element;

        const int active = spline.first_active(element);
        spline.derivs_on_element(active, std::span<const T>(&parameter, 1),
                                 order, spline_derivs);

        T factorial{1};

        for (int k = 0; k <= order; ++k) {
            if (k > 0)
                factorial *= static_cast<T>(k);

            series(k) = spline_derivs[static_cast<std::size_t>(k)].col(0).dot(
                            coefficients[static_cast<std::size_t>(face)]
                                .segment(active, spline.degree() + 1))
                        / factorial;
        }

        // The arc length of the exact curve, with τ along the
        // parametrization where the face sign is 1 and against it otherwise
        tangent_series(curve, projection.elements(point), parameter,
                       std::max(order, 1), dx, dy);
        const Eigen::VectorX<T> inverse_speed = series_power(
            Eigen::VectorX<T>(series_product(dx, dx) + series_product(dy, dy)),
            T{-0.5});
        const Eigen::VectorX<T> along =
            arc_length_derivs(series, inverse_speed, order);

        const T sign = static_cast<T>(boundary.sign(face));
        T factor{1};

        for (int j = 0; j <= order; ++j) {
            derivs(point, j) = factor * along(j);
            factor *= sign;
        }
    }

    return derivs;
}

template BoundaryProjection<double> project_closest_points(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    const PointMatrix<double, 2>&);
template BoundaryProjection<double> project_closest_points(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    const PointMatrix<double, 2>&);
template Eigen::MatrixX<double> curvature_derivs(
    const Boundary<double, 2>&, const BoundaryProjection<double>&, int);
template PointMatrix<double, 2> boundary_interpolation_points(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    int);
template PointMatrix<double, 2> boundary_interpolation_points(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    int);
template Eigen::MatrixX<double> boundary_interpolated_derivs(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    int, const Eigen::VectorX<double>&, const BoundaryProjection<double>&,
    int);
template Eigen::MatrixX<double> boundary_interpolated_derivs(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    int, const Eigen::VectorX<double>&, const BoundaryProjection<double>&,
    int);

} // namespace iguana
