/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <array>
#include <vector>

#include "iguana/iguana.hpp"

namespace py = pybind11;

using iguana::BSpline;
using iguana::Patch;
using iguana::TensorBSpline;

using TrivariateBSpline = TensorBSpline<double, 3>;
using VolumePatch = Patch<double, 3>;

PYBIND11_MODULE(cpp, module)
{
    module.attr("__version__") = IGUANA_VERSION;

    py::class_<BSpline<double>>(module, "BSpline")
        .def(py::init<int, std::vector<double>>(),
             py::arg("degree"), py::arg("knots"));

    py::class_<TrivariateBSpline>(module, "TrivariateBSpline")
        .def(py::init<std::array<BSpline<double>, 3>>(), py::arg("axes"));

    py::class_<VolumePatch>(module, "VolumePatch")
        .def(py::init<TrivariateBSpline, iguana::PointMatrix<double>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("coefficients", &VolumePatch::coefficients);
}
