/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary_projection.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/nurbs/bezier_points.hpp"
#include "iguana/geometry/nurbs/curve_projection.hpp"
#include "iguana/grid/knot_vector.hpp"
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

template BoundaryProjection<double> project_closest_points(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&,
    const PointMatrix<double, 2>&);
template BoundaryProjection<double> project_closest_points(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&,
    const PointMatrix<double, 2>&);
template Eigen::MatrixX<double> curvature_derivs(
    const Boundary<double, 2>&, const BoundaryProjection<double>&, int);

} // namespace iguana
