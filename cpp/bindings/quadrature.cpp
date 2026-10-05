/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"
#include "iguana/utils/multi_index.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using DomainQuadrature2d = DomainQuadrature<double, 2>;
using DomainQuadrature3d = DomainQuadrature<double, 3>;
using BoundaryQuadrature2d = BoundaryQuadrature<double, 2>;
using PlanarPatch = Patch<TensorBSpline<double, 2>, 2>;

/// @brief Fills the cells of one type with a Gauss-Legendre rule, which
///        Python never handles itself
template<std::size_t d>
void fill_gauss_legendre(DomainQuadrature<double, d>& quadrature,
                         const HierarchicalGrid<double, d>& grid,
                         const EmbeddedDomain<double, d>& domain,
                         CellType cell_type,
                         const std::array<int, d>& num_points)
{
    quadrature.fill(grid, domain, cell_type,
                    GaussLegendre<double, d>(num_points));
}

/// @brief Fills the cells of one type with rules fitted to a domain, closed
///        by facets in parameter space: segments with the domain on their
///        left, or triangles counterclockwise seen from outside
template<std::size_t d>
void fill_moment_fitting(DomainQuadrature<double, d>& quadrature,
                         const HierarchicalGrid<double, d>& grid,
                         const EmbeddedDomain<double, d>& domain,
                         CellType cell_type, const Eigen::MatrixXd& vertices,
                         const Eigen::MatrixXi& facets, int order)
{
    quadrature.fill(grid, domain, cell_type,
                    MomentFitting<double, d>(vertices, facets, order));
}

/// @brief Element of a knot vector holding a parameter, the last one that
///        starts at or before it
///
/// @pre @p parameter lies in the parametric domain of @p knots
int element_holding(const KnotVector<double>& knots, double parameter)
{
    int first = 0;
    int last = knots.num_elements() - 1;

    while (first < last) {
        const int middle = (first + last + 1) / 2;

        if (knots.element_start(middle) <= parameter)
            first = middle;
        else
            last = middle - 1;
    }

    return first;
}

/**
 * @brief Walks the elements a quadrature holds, giving each the element of
 *        the patch holding it and the range of its points
 *
 * Every element of the grid lies inside an element of the patch, the one
 * holding the parameters at which it starts, whose functions map its
 * points. The visit receives the first active function of that element of
 * the patch in each direction, its index, and the first point and the
 * number of points of the element of the grid
 *
 * @throws std::invalid_argument If the coarse level of the grid does not
 *         have the knots of the patch, or if an element of the quadrature
 *         lies outside the grid
 */
template<typename Basis, std::size_t n, typename Visit>
void walk_held_elements(
    const DomainQuadrature<double, Basis::dimension>& quadrature,
    const Patch<Basis, n>& patch,
    const HierarchicalGrid<double, Basis::dimension>& grid, Visit&& visit)
{
    constexpr std::size_t d = Basis::dimension;
    const Basis& basis = patch.basis();

    for (std::size_t direction = 0; direction < d; ++direction) {
        const KnotVector<double>& coarse = grid.level(0).knots(direction);
        const KnotVector<double>& knots = basis.grid().knots(direction);

        if (coarse.degree() != knots.degree() ||
            coarse.values() != knots.values())
            throw std::invalid_argument("DomainQuadrature: "
                                        "the grid must lie on the knots of "
                                        "the patch");
    }

    const Eigen::VectorXi& offsets = quadrature.offsets();

    // Position of each element among the held ones, or -1 if not held
    std::vector<int> held(grid.num_elements(), -1);

    for (int position = 0; position < quadrature.num_elements(); ++position) {
        const int element = quadrature.elements()(position);

        if (element < 0 || element >= grid.num_elements())
            throw std::invalid_argument("DomainQuadrature: "
                                        "the elements must lie in the grid");

        held[element] = position;
    }

    for (const HierarchicalGridIterator<double, d>& element : grid) {
        const int position = held[element.index()];

        if (position < 0)
            continue;

        // The element of the patch holding it, with the first direction
        // running fastest
        std::array<int, d> first_active{};
        int coarse_element = 0;
        int stride = 1;

        for (std::size_t direction = 0; direction < d; ++direction) {
            const KnotVector<double>& knots = basis.grid().knots(direction);
            const int axis_element =
                element_holding(knots, element.start()[direction]);

            first_active[direction] =
                knots.element_span(axis_element) - knots.degree();
            coarse_element += axis_element * stride;
            stride *= knots.num_elements();
        }

        const int first = offsets(position);
        const int count = offsets(position + 1) - first;

        visit(first_active, coarse_element, first, count);
    }
}

/**
 * @brief Positions of the points in physical space, the only form Python
 *        needs, as it draws them
 *
 * @throws std::invalid_argument As walk_held_elements() does
 */
template<typename Basis, std::size_t n>
PointMatrix<double, n> positions(
    const DomainQuadrature<double, Basis::dimension>& quadrature,
    const Patch<Basis, n>& patch,
    const HierarchicalGrid<double, Basis::dimension>& grid)
{
    constexpr std::size_t d = Basis::dimension;
    PointMatrix<double, n> result(quadrature.num_points(), n);

    // Buffers reused over the elements
    Eigen::MatrixXd parameters;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    PointMatrix<double, n> element_positions;

    walk_held_elements(
        quadrature, patch, grid,
        [&](const std::array<int, d>& first_active, int element, int first,
            int count) {
            parameters = quadrature.points().middleRows(first, count);
            patch.basis().eval_on_element(first_active, parameters, values);
            patch.basis().active_on_element(element, actives);
            patch.position_on_element(actives, values, element_positions);

            result.middleRows(first, count) = element_positions;
        });

    return result;
}

/**
 * @brief Weights of the points in physical space, each parametric weight
 *        times the measure of the patch at its point
 *
 * @throws std::invalid_argument As walk_held_elements() does
 */
template<typename Basis, std::size_t n>
Eigen::VectorXd physical_weights(
    const DomainQuadrature<double, Basis::dimension>& quadrature,
    const Patch<Basis, n>& patch,
    const HierarchicalGrid<double, Basis::dimension>& grid)
{
    constexpr std::size_t d = Basis::dimension;
    Eigen::VectorXd result = quadrature.weights();

    // Buffers reused over the elements
    Eigen::MatrixXd parameters;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    Eigen::VectorXd measures;
    std::array<Eigen::MatrixXd, d> gradients;
    std::array<PointMatrix<double, n>, d> tangents;

    walk_held_elements(
        quadrature, patch, grid,
        [&](const std::array<int, d>& first_active, int element, int first,
            int count) {
            parameters = quadrature.points().middleRows(first, count);
            patch.basis().grad_on_element(first_active, parameters, values,
                                          gradients);
            patch.basis().active_on_element(element, actives);
            patch.tangent_on_element(actives, gradients, tangents);
            Patch<Basis, n>::measure_on_element(tangents, measures);

            result.segment(first, count).array() *= measures.array();
        });

    return result;
}

/// @brief Places a Gauss-Legendre rule on every piece of an embedded
///        boundary, a rule Python never handles itself
BoundaryQuadrature2d boundary_gauss_legendre(
    const EmbeddedBoundary<double, 2>& boundary, int num_points)
{
    return {boundary, GaussLegendre<double, 1>(num_points)};
}

/**
 * @brief Walks the elements a boundary quadrature holds, giving each the
 *        first active function of the patch in each direction, its index,
 *        and its first point and number of points
 *
 * @throws std::invalid_argument If an element of the quadrature lies
 *         outside the grid of the patch
 *
 * @pre The boundary of the quadrature was divided over the grid of the
 *      patch
 */
template<typename Visit>
void walk_held_elements(const BoundaryQuadrature2d& quadrature,
                        const PlanarPatch& patch, Visit&& visit)
{
    const TensorBSpline<double, 2>& basis = patch.basis();
    const TensorGrid<double, 2>& grid = basis.grid();
    const std::array<int, 2> counts{grid.knots(0).num_elements(),
                                    grid.knots(1).num_elements()};
    const Eigen::VectorXi& offsets = quadrature.offsets();

    for (int position = 0; position < quadrature.num_elements(); ++position) {
        const int element = quadrature.elements()(position);

        if (element < 0 || element >= grid.num_elements())
            throw std::invalid_argument("BoundaryQuadrature: "
                                        "the elements must lie in the grid "
                                        "of the patch");

        const std::array<int, 2> axis_elements = unflatten(element, counts);
        const std::array<int, 2> first_active{
            basis.axis(0).first_active(axis_elements[0]),
            basis.axis(1).first_active(axis_elements[1])};

        const int first = offsets(position);
        const int count = offsets(position + 1) - first;

        visit(first_active, element, first, count);
    }
}

/**
 * @brief Positions of the points of a boundary quadrature in the plane
 *
 * @throws std::invalid_argument As walk_held_elements() does
 */
PointMatrix<double, 2> boundary_positions(
    const BoundaryQuadrature2d& quadrature, const PlanarPatch& patch)
{
    PointMatrix<double, 2> result(quadrature.num_points(), 2);

    // Buffers reused over the elements
    Eigen::MatrixXd parameters;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    PointMatrix<double, 2> element_positions;

    walk_held_elements(
        quadrature, patch,
        [&](const std::array<int, 2>& first_active, int element, int first,
            int count) {
            parameters = quadrature.points().middleRows(first, count);
            patch.basis().eval_on_element(first_active, parameters, values);
            patch.basis().active_on_element(element, actives);
            patch.position_on_element(actives, values, element_positions);

            result.middleRows(first, count) = element_positions;
        });

    return result;
}

/**
 * @brief Physical normal of each point of a boundary quadrature scaled by
 *        its physical weight, by Nanson's formula |det J| J^-T m w
 *
 * @throws std::invalid_argument As walk_held_elements() does
 */
PointMatrix<double, 2> scaled_normals(const BoundaryQuadrature2d& quadrature,
                                      const PlanarPatch& patch)
{
    PointMatrix<double, 2> result(quadrature.num_points(), 2);

    // Buffers reused over the elements
    Eigen::MatrixXd parameters;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    std::array<Eigen::MatrixXd, 2> gradients;
    std::array<PointMatrix<double, 2>, 2> tangents;

    walk_held_elements(
        quadrature, patch,
        [&](const std::array<int, 2>& first_active, int element, int first,
            int count) {
            parameters = quadrature.points().middleRows(first, count);
            patch.basis().grad_on_element(first_active, parameters, values,
                                          gradients);
            patch.basis().active_on_element(element, actives);
            patch.tangent_on_element(actives, gradients, tangents);

            for (int point = 0; point < count; ++point) {
                const int row = first + point;

                // The tangents are the columns of the Jacobian
                Eigen::Matrix2d jacobian;
                jacobian << tangents[0].row(point).transpose(),
                            tangents[1].row(point).transpose();

                result.row(row) =
                    std::abs(jacobian.determinant()) *
                    quadrature.weights()(row) *
                    (jacobian.inverse().transpose() *
                     quadrature.normals().row(row).transpose())
                        .transpose();
            }
        });

    return result;
}

/**
 * @brief Weights of the points of a boundary quadrature in the plane, the
 *        lengths of their scaled normals
 *
 * @throws std::invalid_argument As walk_held_elements() does
 */
Eigen::VectorXd physical_boundary_weights(
    const BoundaryQuadrature2d& quadrature, const PlanarPatch& patch)
{
    return scaled_normals(quadrature, patch).rowwise().norm();
}

/**
 * @brief Unit normals of the points of a boundary quadrature in the plane,
 *        pointing out of the domain, the directions of their scaled normals
 *
 * @throws std::invalid_argument As walk_held_elements() does
 */
PointMatrix<double, 2> physical_boundary_normals(
    const BoundaryQuadrature2d& quadrature, const PlanarPatch& patch)
{
    return scaled_normals(quadrature, patch).rowwise().normalized();
}

} // namespace

void quadrature(py::module_& module)
{
    // The weights are copied, as filling reallocates them
    constexpr py::return_value_policy copy = py::return_value_policy::copy;

    py::class_<DomainQuadrature2d>(module, "DomainQuadrature2d")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<2>,
             py::arg("grid"), py::arg("domain"), py::arg("cell_type"),
             py::arg("num_points"))
        .def("fill_moment_fitting", &fill_moment_fitting<2>,
             py::arg("grid"), py::arg("domain"), py::arg("cell_type"),
             py::arg("vertices"), py::arg("facets"), py::arg("order"))
        .def_property_readonly("num_elements",
                               &DomainQuadrature2d::num_elements)
        .def_property_readonly("num_points", &DomainQuadrature2d::num_points)
        .def_property_readonly("weights", &DomainQuadrature2d::weights, copy)
        .def("positions", &positions<TensorBSpline<double, 2>, 2>,
             py::arg("patch"), py::arg("grid"))
        .def("positions", &positions<TensorBSpline<double, 2>, 3>,
             py::arg("patch"), py::arg("grid"))
        .def("positions", &positions<TensorNURBS<double, 2>, 3>,
             py::arg("patch"), py::arg("grid"))
        .def("physical_weights", &physical_weights<TensorBSpline<double, 2>, 2>,
             py::arg("patch"), py::arg("grid"))
        .def("physical_weights", &physical_weights<TensorBSpline<double, 2>, 3>,
             py::arg("patch"), py::arg("grid"))
        .def("physical_weights", &physical_weights<TensorNURBS<double, 2>, 3>,
             py::arg("patch"), py::arg("grid"));

    py::class_<DomainQuadrature3d>(module, "DomainQuadrature3d")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<3>,
             py::arg("grid"), py::arg("domain"), py::arg("cell_type"),
             py::arg("num_points"))
        .def("fill_moment_fitting", &fill_moment_fitting<3>,
             py::arg("grid"), py::arg("domain"), py::arg("cell_type"),
             py::arg("vertices"), py::arg("facets"), py::arg("order"))
        .def_property_readonly("num_elements",
                               &DomainQuadrature3d::num_elements)
        .def_property_readonly("num_points", &DomainQuadrature3d::num_points)
        .def_property_readonly("weights", &DomainQuadrature3d::weights, copy)
        .def("positions", &positions<TensorBSpline<double, 3>, 3>,
             py::arg("patch"), py::arg("grid"))
        .def("physical_weights", &physical_weights<TensorBSpline<double, 3>, 3>,
             py::arg("patch"), py::arg("grid"));

    py::class_<BoundaryQuadrature2d>(module, "BoundaryQuadrature2d")
        .def(py::init(&boundary_gauss_legendre), py::arg("boundary"),
             py::arg("num_points"))
        .def_property_readonly("num_elements",
                               &BoundaryQuadrature2d::num_elements)
        .def_property_readonly("num_points", &BoundaryQuadrature2d::num_points)
        .def_property_readonly("faces", &BoundaryQuadrature2d::faces)
        .def("positions", &boundary_positions, py::arg("patch"))
        .def("physical_weights", &physical_boundary_weights, py::arg("patch"))
        .def("physical_normals", &physical_boundary_normals,
             py::arg("patch"));
}

} // namespace iguana::bindings
