/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

void basis(py::module_& module)
{
    py::class_<BSpline<double>>(module, "BSpline")
        .def(py::init<int, std::vector<double>>(),
             py::arg("degree"), py::arg("knots"))
        .def_property_readonly("degree", &BSpline<double>::degree)
        .def_property_readonly("knots", &BSpline<double>::knots);

    py::class_<TensorBSpline<double, 1>>(module, "UnivariateBSpline")
        .def("axis", &TensorBSpline<double, 1>::axis);

    py::class_<TensorBSpline<double, 2>>(module, "BivariateBSpline")
        .def("axis", &TensorBSpline<double, 2>::axis);

    py::class_<TensorBSpline<double, 3>>(module, "TrivariateBSpline")
        .def(py::init<std::array<BSpline<double>, 3>>(), py::arg("axes"))
        .def("axis", &TensorBSpline<double, 3>::axis);
}

} // namespace iguana::bindings
