/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <iterator>
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
                         const TensorBSpline<double, d>& basis,
                         const Embedding<double, d>& embedding,
                         CellType cell_type,
                         const std::array<int, d>& num_points)
{
    quadrature.fill(basis, embedding, cell_type,
                    GaussLegendre<double, d>(num_points));
}

/// @brief Fills the cells of one type with rules fitted to a solid, given
///        as a closed triangle mesh in parameter space
void fill_moment_fitting(VolumeQuadrature& quadrature,
                         const TensorBSpline<double, 3>& basis,
                         const Embedding<double, 3>& embedding,
                         CellType cell_type, const Eigen::MatrixXd& vertices,
                         const Eigen::MatrixXi& triangles, int order)
{
    quadrature.fill(basis, embedding, cell_type,
                    MomentFitting<double, 3>(vertices, triangles, order));
}

/**
 * @brief Positions of the points in physical space, the only form Python
 *        needs, as it draws them
 *
 * @throws std::invalid_argument If an element of the quadrature lies
 *         outside the basis of the patch
 */
template<std::size_t d>
PointMatrix<double> positions(const DomainQuadrature<double, d>& quadrature,
                              const Patch<double, d>& patch)
{
    const TensorBSpline<double, d>& basis = patch.basis();
    const Eigen::VectorXi& offsets = quadrature.offsets();

    // Position of each element among the held ones, or -1 if not held
    std::vector<int> held(basis.num_elements(), -1);

    for (int position = 0; position < quadrature.num_elements(); ++position) {
        const int element = quadrature.elements()(position);

        if (element < 0 || element >= basis.num_elements())
            throw std::invalid_argument("DomainQuadrature: "
                                        "the elements must lie in the "
                                        "basis");

        held[element] = position;
    }

    PointMatrix<double> result(quadrature.num_points(), 3);

    // Buffers reused over the elements
    Eigen::MatrixXd parameters;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    PointMatrix<double> element_positions;

    for (TensorDomainIterator<double, d> element(basis);
         element != std::default_sentinel; ++element) {
        const int position = held[element.index()];

        if (position < 0)
            continue;

        const int first = offsets(position);
        const int count = offsets(position + 1) - first;

        parameters = quadrature.points().middleRows(first, count);
        basis.eval_on_element(element.first_active(), parameters, values);
        basis.active_on_element(element.index(), actives);
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
             py::arg("basis"), py::arg("embedding"), py::arg("cell_type"),
             py::arg("num_points"))
        .def_property_readonly("num_elements",
                               &SurfaceQuadrature::num_elements)
        .def_property_readonly("num_points", &SurfaceQuadrature::num_points)
        .def_property_readonly("weights", &SurfaceQuadrature::weights, copy)
        .def("positions", &positions<2>, py::arg("patch"));

    py::class_<VolumeQuadrature>(module, "VolumeQuadrature")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<3>,
             py::arg("basis"), py::arg("embedding"), py::arg("cell_type"),
             py::arg("num_points"))
        .def("fill_moment_fitting", &fill_moment_fitting, py::arg("basis"),
             py::arg("embedding"), py::arg("cell_type"), py::arg("vertices"),
             py::arg("triangles"), py::arg("order"))
        .def_property_readonly("num_elements",
                               &VolumeQuadrature::num_elements)
        .def_property_readonly("num_points", &VolumeQuadrature::num_points)
        .def_property_readonly("weights", &VolumeQuadrature::weights, copy)
        .def("positions", &positions<3>, py::arg("patch"));
}

} // namespace iguana::bindings
