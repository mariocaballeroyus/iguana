/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>

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
                         const TensorDomain<double, d>& domain,
                         CellType cell_type,
                         const std::array<int, d>& num_points)
{
    quadrature.fill(domain, cell_type, GaussLegendre<double, d>(num_points));
}

} // namespace

void quadrature(py::module_& module)
{
    py::class_<SurfaceQuadrature>(module, "SurfaceQuadrature")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<2>,
             py::arg("domain"), py::arg("cell_type"), py::arg("num_points"))
        .def_property_readonly("num_elements",
                               &SurfaceQuadrature::num_elements)
        .def_property_readonly("num_points", &SurfaceQuadrature::num_points);

    py::class_<VolumeQuadrature>(module, "VolumeQuadrature")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<3>,
             py::arg("domain"), py::arg("cell_type"), py::arg("num_points"))
        .def_property_readonly("num_elements",
                               &VolumeQuadrature::num_elements)
        .def_property_readonly("num_points", &VolumeQuadrature::num_points);
}

} // namespace iguana::bindings
