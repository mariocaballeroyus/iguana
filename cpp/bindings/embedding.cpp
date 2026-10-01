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

using EmbeddedDomain2d = EmbeddedDomain<double, 2>;
using EmbeddedDomain3d = EmbeddedDomain<double, 3>;

} // namespace

void embedding(py::module_& module)
{
    py::enum_<CellType>(module, "CellType")
        .value("outside", CellType::outside)
        .value("inside", CellType::inside)
        .value("cut", CellType::cut);

    py::class_<EmbeddedDomain2d>(module, "EmbeddedDomain2d")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));

    py::class_<EmbeddedDomain3d>(module, "EmbeddedDomain3d")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));
}

} // namespace iguana::bindings
