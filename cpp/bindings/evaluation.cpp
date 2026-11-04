/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

void evaluation(py::module_& module)
{
    using Basis2d = TensorBSpline<double, 2>;

    module.def("trace_at_points", &trace_at_points<Basis2d>, py::arg("space"),
               py::arg("patch"), py::arg("trace"), py::arg("coefficients"),
               py::arg("points"));
    module.def("flux_at_points", &flux_at_points<Basis2d>, py::arg("space"),
               py::arg("patch"), py::arg("flux"), py::arg("coefficients"),
               py::arg("points"), py::arg("normals"));
}

} // namespace iguana::bindings
