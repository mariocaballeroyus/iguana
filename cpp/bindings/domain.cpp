/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using SurfaceDomain = TensorDomain<double, 2>;
using VolumeDomain = TensorDomain<double, 3>;

} // namespace

void domain(py::module_& module)
{
    py::enum_<CellType>(module, "CellType")
        .value("outside", CellType::outside)
        .value("inside", CellType::inside)
        .value("cut", CellType::cut);

    py::class_<SurfaceDomain>(module, "SurfaceDomain")
        .def(py::init<Patch<double, 2>>(), py::arg("patch"))
        .def(py::init<Patch<double, 2>, std::vector<CellType>>(),
             py::arg("patch"), py::arg("cell_types"))
        .def_property_readonly("num_elements", &SurfaceDomain::num_elements)
        .def("cell_type", &SurfaceDomain::cell_type, py::arg("element"));

    py::class_<VolumeDomain>(module, "VolumeDomain")
        .def(py::init<Patch<double, 3>>(), py::arg("patch"))
        .def(py::init<Patch<double, 3>, std::vector<CellType>>(),
             py::arg("patch"), py::arg("cell_types"))
        .def_property_readonly("num_elements", &VolumeDomain::num_elements)
        .def("cell_type", &VolumeDomain::cell_type, py::arg("element"));
}

} // namespace iguana::bindings
