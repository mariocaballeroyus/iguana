/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using VolumePatch = Patch<double, 3>;

} // namespace

void patch(py::module_& module)
{
    py::class_<VolumePatch>(module, "VolumePatch")
        .def(py::init<TensorBSpline<double, 3>, PointMatrix<double>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("coefficients", &VolumePatch::coefficients);
}

} // namespace iguana::bindings
