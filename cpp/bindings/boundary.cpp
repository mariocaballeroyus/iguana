/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <vector>

#include <pybind11/eigen.h>
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
using BoundaryProjection2d = BoundaryProjection<double>;
using PoissonShift2d = PoissonShift<double>;
using StokesShift2d = StokesShift<double>;

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

    // The projection of the points of a planar patch, which the conditions
    // of the shifted boundary method pass to the assembly, with the shift
    // weights a condition rewritten with an equation sets, and the further
    // weights, set as a whole list, of a trace that takes another field
    py::class_<BoundaryProjection2d>(module, "BoundaryProjection2d")
        .def_readonly("positions", &BoundaryProjection2d::positions)
        .def_readwrite("shift_weights", &BoundaryProjection2d::shift_weights)
        .def_readwrite("coupled_weights",
                       &BoundaryProjection2d::coupled_weights);

    module.def("project_closest_points",
               &project_closest_points<TensorBSpline<double, 2>>,
               py::arg("patch"), py::arg("boundary"), py::arg("points"));

    // The shift of a Dirichlet condition rewritten with the Poisson
    // equation, at the points of a projection
    py::class_<PoissonShift2d>(module, "PoissonShift2d")
        .def_readonly("weights", &PoissonShift2d::weights)
        .def_readonly("source_weights", &PoissonShift2d::source_weights);

    module.def("poisson_shift", &poisson_shift<TensorBSpline<double, 2>>,
               py::arg("patch"), py::arg("projection"), py::arg("order"));

    // The shift of the velocity of a Dirichlet condition rewritten with the
    // Stokes equation, at the points of a projection
    py::class_<StokesShift2d>(module, "StokesShift2d")
        .def_readonly("weights", &StokesShift2d::weights)
        .def_readonly("pressure_weights", &StokesShift2d::pressure_weights)
        .def_readonly("source_weights", &StokesShift2d::source_weights);

    module.def("stokes_shift", &stokes_shift<TensorBSpline<double, 2>>,
               py::arg("patch"), py::arg("projection"), py::arg("order"),
               py::arg("viscosity"));

    // Data known at points, interpolated in the basis of a patch, with the
    // derivatives of the interpolant at points
    module.def("interpolation_points", &interpolation_points<double>,
               py::arg("patch"));
    module.def("interpolated_derivs", &interpolated_derivs<double>,
               py::arg("patch"), py::arg("values"), py::arg("points"),
               py::arg("order"));

    py::class_<Boundary3d>(module, "Boundary3d")
        .def(py::init<std::vector<Boundary3d::Face>, std::vector<int>>(),
             py::arg("faces"), py::arg("signs"))
        .def_property_readonly("num_faces", &Boundary3d::num_faces)
        .def_property_readonly("faces", &faces<3>)
        .def_property_readonly("signs", &signs<3>);
}

} // namespace iguana::bindings
