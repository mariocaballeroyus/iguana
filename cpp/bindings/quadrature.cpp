/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using SurfaceQuadrature = DomainQuadrature<double, 2>;
using VolumeQuadrature = DomainQuadrature<double, 3>;

/// @brief Fills the cells of one type with a Gauss-Legendre rule, which
///        Python never handles itself
template<std::size_t d>
void fill_gauss_legendre(DomainQuadrature<double, d>& quadrature,
                         const HierarchicalGrid<double, d>& grid,
                         const Embedding<double, d>& embedding,
                         CellType cell_type,
                         const std::array<int, d>& num_points)
{
    quadrature.fill(grid, embedding, cell_type,
                    GaussLegendre<double, d>(num_points));
}

/// @brief Fills the cells of one type with rules fitted to a domain, closed
///        by facets in parameter space: segments with the domain on their
///        left, or triangles counterclockwise seen from outside
template<std::size_t d>
void fill_moment_fitting(DomainQuadrature<double, d>& quadrature,
                         const HierarchicalGrid<double, d>& grid,
                         const Embedding<double, d>& embedding,
                         CellType cell_type, const Eigen::MatrixXd& vertices,
                         const Eigen::MatrixXi& facets, int order)
{
    quadrature.fill(grid, embedding, cell_type,
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
 * @brief Positions of the points in physical space, the only form Python
 *        needs, as it draws them
 *
 * Every element of the grid lies inside an element of the patch, the one
 * holding the parameters at which it starts, whose functions map its points
 *
 * @throws std::invalid_argument If the coarse level of the grid does not
 *         have the knots of the patch, or if an element of the quadrature
 *         lies outside the grid
 */
template<typename Basis, std::size_t n>
PointMatrix<double, n> positions(
    const DomainQuadrature<double, Basis::dimension>& quadrature,
    const Patch<Basis, n>& patch,
    const HierarchicalGrid<double, Basis::dimension>& grid)
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

    PointMatrix<double, n> result(quadrature.num_points(), n);

    // Buffers reused over the elements
    Eigen::MatrixXd parameters;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    PointMatrix<double, n> element_positions;

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

        parameters = quadrature.points().middleRows(first, count);
        basis.eval_on_element(first_active, parameters, values);
        basis.active_on_element(coarse_element, actives);
        patch.position_on_element(actives, values, element_positions);

        result.middleRows(first, count) = element_positions;
    }

    return result;
}

} // namespace

void quadrature(py::module_& module)
{
    // The weights are copied, as filling reallocates them
    constexpr py::return_value_policy copy = py::return_value_policy::copy;

    py::class_<SurfaceQuadrature>(module, "SurfaceQuadrature")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<2>,
             py::arg("grid"), py::arg("embedding"), py::arg("cell_type"),
             py::arg("num_points"))
        .def("fill_moment_fitting", &fill_moment_fitting<2>,
             py::arg("grid"), py::arg("embedding"), py::arg("cell_type"),
             py::arg("vertices"), py::arg("facets"), py::arg("order"))
        .def_property_readonly("num_elements",
                               &SurfaceQuadrature::num_elements)
        .def_property_readonly("num_points", &SurfaceQuadrature::num_points)
        .def_property_readonly("weights", &SurfaceQuadrature::weights, copy)
        .def("positions", &positions<TensorBSpline<double, 2>, 2>,
             py::arg("patch"), py::arg("grid"))
        .def("positions", &positions<TensorBSpline<double, 2>, 3>,
             py::arg("patch"), py::arg("grid"))
        .def("positions", &positions<TensorNURBS<double, 2>, 3>,
             py::arg("patch"), py::arg("grid"));

    py::class_<VolumeQuadrature>(module, "VolumeQuadrature")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<3>,
             py::arg("grid"), py::arg("embedding"), py::arg("cell_type"),
             py::arg("num_points"))
        .def("fill_moment_fitting", &fill_moment_fitting<3>,
             py::arg("grid"), py::arg("embedding"), py::arg("cell_type"),
             py::arg("vertices"), py::arg("facets"), py::arg("order"))
        .def_property_readonly("num_elements",
                               &VolumeQuadrature::num_elements)
        .def_property_readonly("num_points", &VolumeQuadrature::num_points)
        .def_property_readonly("weights", &VolumeQuadrature::weights, copy)
        .def("positions", &positions<TensorBSpline<double, 3>, 3>,
             py::arg("patch"), py::arg("grid"));
}

} // namespace iguana::bindings
