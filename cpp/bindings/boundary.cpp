/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using Boundary2d = Boundary<double, 2>;
using Boundary3d = Boundary<double, 3>;

/// @brief Faces of a boundary, copied so that Python never holds an index
///        outside them
template<std::size_t n>
std::vector<typename Boundary<double, n>::Face> faces(
    const Boundary<double, n>& boundary)
{
    std::vector<typename Boundary<double, n>::Face> result;

    for (int face = 0; face < boundary.num_faces(); ++face)
        result.push_back(boundary.face(face));

    return result;
}

/// @brief Sign of each face, in the order of the faces
template<std::size_t n>
std::vector<int> signs(const Boundary<double, n>& boundary)
{
    std::vector<int> result;

    for (int face = 0; face < boundary.num_faces(); ++face)
        result.push_back(boundary.sign(face));

    return result;
}

} // namespace

void boundary(py::module_& module)
{
    py::class_<Boundary2d>(module, "Boundary2d")
        .def(py::init<std::vector<Boundary2d::Face>, std::vector<int>>(),
             py::arg("faces"), py::arg("signs"))
        .def_property_readonly("num_faces", &Boundary2d::num_faces)
        .def_property_readonly("faces", &faces<2>)
        .def_property_readonly("signs", &signs<2>);

    py::class_<Boundary3d>(module, "Boundary3d")
        .def(py::init<std::vector<Boundary3d::Face>, std::vector<int>>(),
             py::arg("faces"), py::arg("signs"))
        .def_property_readonly("num_faces", &Boundary3d::num_faces)
        .def_property_readonly("faces", &faces<3>)
        .def_property_readonly("signs", &signs<3>);
}

} // namespace iguana::bindings
