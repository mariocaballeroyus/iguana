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

using SurfaceEmbedding = Embedding<double, 2>;
using VolumeEmbedding = Embedding<double, 3>;

} // namespace

void embedding(py::module_& module)
{
    py::enum_<CellType>(module, "CellType")
        .value("outside", CellType::outside)
        .value("inside", CellType::inside)
        .value("cut", CellType::cut);

    py::class_<SurfaceEmbedding>(module, "SurfaceEmbedding")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));

    py::class_<VolumeEmbedding>(module, "VolumeEmbedding")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));
}

} // namespace iguana::bindings
