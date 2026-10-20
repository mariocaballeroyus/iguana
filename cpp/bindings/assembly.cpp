/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "iguana/iguana.hpp"
#include "petsc_caster.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using Trace2d = Trace<TensorBSpline<double, 2>, 2>;
using Flux2d = Flux<TensorBSpline<double, 2>, 2>;
using PoissonElement2d = PoissonElement<TensorBSpline<double, 2>, 2>;
using StokesElement2d = StokesElement<TensorBSpline<double, 2>, 2>;
using StokesStabilizedElement2d =
    StokesStabilizedElement<TensorBSpline<double, 2>, 2>;
using PenaltyCondition2d = PenaltyCondition<TensorBSpline<double, 2>, 2>;
using NeumannCondition2d = NeumannCondition<TensorBSpline<double, 2>, 2>;
using NitscheCondition2d = NitscheCondition<TensorBSpline<double, 2>, 2>;
using SkeletonNitscheCondition2d =
    SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>;
using SkeletonGhostPenalty2d =
    SkeletonGhostPenalty<TensorBSpline<double, 2>, 2>;
using BoundaryProjection2d = BoundaryProjection<double>;
using QuadratureBoundary2d = QuadratureBoundary<double, 2>;
using QuadratureSkeleton2d = QuadratureSkeleton<double, 2>;
using Assembler2d = Assembler<TensorBSpline<double, 2>, 2>;
using Form2d = Form<TensorBSpline<double, 2>, 2>;
using Data = Form2d::Data;

/// @brief Integral of a ghost penalty across faces, the one overload of
///        add_skeleton_integral that is not a template
using AddGhostPenalty2d = void (Form2d::*)(const SkeletonGhostPenalty2d&,
                                           const QuadratureSkeleton2d&);

} // namespace

void assembly(py::module_& module)
{
    // The bases of the traces and fluxes, so that a condition takes those
    // of any element
    py::class_<Trace2d>(module, "Trace2d");
    py::class_<Flux2d>(module, "Flux2d");

    py::class_<PoissonElement2d> poisson_element_2d(module,
                                                    "PoissonElement2d");
    poisson_element_2d.def(py::init<>());

    py::class_<PoissonElement2d::U, Trace2d>(poisson_element_2d, "U")
        .def(py::init<>());

    py::class_<PoissonElement2d::Q, Flux2d>(poisson_element_2d, "Q")
        .def(py::init<>());

    // The Galerkin element, which no problem builds yet, holds the traces
    // and fluxes of both
    py::class_<StokesElement2d> stokes_element_2d(module, "StokesElement2d");

    py::class_<StokesStabilizedElement2d, StokesElement2d>(
        module, "StokesStabilizedElement2d")
        .def(py::init<double, double>(), py::arg("viscosity"),
             py::arg("stabilization"));

    py::class_<StokesElement2d::U, Trace2d>(stokes_element_2d, "U")
        .def(py::init<std::size_t>(), py::arg("component"));

    py::class_<StokesElement2d::P, Trace2d>(stokes_element_2d, "P")
        .def(py::init<>());

    py::class_<StokesElement2d::T, Flux2d>(stokes_element_2d, "T")
        .def(py::init<double, std::size_t>(), py::arg("viscosity"),
             py::arg("component"));

    // A condition keeps a reference to its trace, which Python keeps alive
    // with it
    py::class_<PenaltyCondition2d>(module, "PenaltyCondition2d")
        .def(py::init<const Trace2d&, double>(), py::arg("trace"),
             py::arg("penalty"), py::keep_alive<1, 2>());

    py::class_<NeumannCondition2d>(module, "NeumannCondition2d")
        .def(py::init<const Trace2d&>(), py::arg("trace"),
             py::keep_alive<1, 2>());

    py::class_<NitscheCondition2d>(module, "NitscheCondition2d")
        .def(py::init<const Trace2d&, const Flux2d&, double>(),
             py::arg("trace"), py::arg("flux"), py::arg("penalty"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>());

    py::class_<SkeletonNitscheCondition2d>(module, "SkeletonNitscheCondition2d")
        .def(py::init<const Trace2d&, const Flux2d&, double>(),
             py::arg("trace"), py::arg("flux"), py::arg("penalty"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>());

    py::class_<SkeletonGhostPenalty2d>(module, "SkeletonGhostPenalty2d")
        .def(py::init<const Trace2d&, double, int>(), py::arg("trace"),
             py::arg("penalty"), py::arg("scaling") = 0,
             py::keep_alive<1, 2>());

    // An assembler keeps references to its space and patch, which Python
    // keeps alive with it, and shares its PETSc system with Python through
    // petsc4py
    py::class_<Assembler2d>(module, "Assembler2d")
        .def(py::init<const FunctionSpace<TensorBSpline<double, 2>>&,
                      const Patch<TensorBSpline<double, 2>, 2>&, bool>(),
             py::arg("space"), py::arg("patch"),
             py::arg("couple_faces") = false, py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def("clear", &Assembler2d::clear)
        .def_property_readonly("stiffness", &Assembler2d::stiffness)
        .def_property_readonly("mass", &Assembler2d::mass)
        .def_property_readonly("load", &Assembler2d::load);

    // A form keeps references to what its integrals read, which Python keeps
    // alive with it. Its data are the arrays themselves, never copies, so
    // that the problems of iguana.petsc change them in place between
    // assemblies
    py::class_<Form2d>(module, "Form2d")
        .def(py::init<>())
        .def("add_domain_integral",
             &Form2d::add_domain_integral<PoissonElement2d>,
             py::arg("element"), py::arg("quadrature"),
             py::arg("source").noconvert(),
             py::arg("cell_weights").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>(),
             py::keep_alive<1, 5>())
        .def("add_domain_integral",
             &Form2d::add_domain_integral<StokesStabilizedElement2d>,
             py::arg("element"), py::arg("quadrature"),
             py::arg("source").noconvert(),
             py::arg("cell_weights").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>(),
             py::keep_alive<1, 5>())
        .def("add_mass_integral",
             &Form2d::add_mass_integral<PoissonElement2d>,
             py::arg("element"), py::arg("quadrature"),
             py::arg("cell_weights").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("add_mass_integral",
             &Form2d::add_mass_integral<StokesStabilizedElement2d>,
             py::arg("element"), py::arg("quadrature"),
             py::arg("cell_weights").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("add_boundary_integral",
             py::overload_cast<const PenaltyCondition2d&,
                               const QuadratureBoundary2d&, Data>(
                 &Form2d::add_boundary_integral<PenaltyCondition2d>),
             py::arg("condition"), py::arg("quadrature"),
             py::arg("data").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("add_boundary_integral",
             py::overload_cast<const NeumannCondition2d&,
                               const QuadratureBoundary2d&, Data>(
                 &Form2d::add_boundary_integral<NeumannCondition2d>),
             py::arg("condition"), py::arg("quadrature"),
             py::arg("data").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("add_boundary_integral",
             py::overload_cast<const NitscheCondition2d&,
                               const QuadratureBoundary2d&, Data>(
                 &Form2d::add_boundary_integral<NitscheCondition2d>),
             py::arg("condition"), py::arg("quadrature"),
             py::arg("data").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("add_boundary_integral",
             py::overload_cast<const PenaltyCondition2d&,
                               const QuadratureBoundary2d&,
                               const BoundaryProjection2d&, int, Data>(
                 &Form2d::add_boundary_integral<PenaltyCondition2d>),
             py::arg("condition"), py::arg("quadrature"),
             py::arg("projection"), py::arg("order"),
             py::arg("data").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>(),
             py::keep_alive<1, 6>())
        .def("add_boundary_integral",
             py::overload_cast<const NitscheCondition2d&,
                               const QuadratureBoundary2d&,
                               const BoundaryProjection2d&, int, Data>(
                 &Form2d::add_boundary_integral<NitscheCondition2d>),
             py::arg("condition"), py::arg("quadrature"),
             py::arg("projection"), py::arg("order"),
             py::arg("data").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>(),
             py::keep_alive<1, 6>())
        .def("add_skeleton_integral",
             &Form2d::add_skeleton_integral<SkeletonNitscheCondition2d>,
             py::arg("condition"), py::arg("quadrature"),
             py::arg("projection"), py::arg("order"),
             py::arg("data").noconvert(),
             py::arg("cell_weights").noconvert(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>(),
             py::keep_alive<1, 6>(), py::keep_alive<1, 7>())
        .def("add_skeleton_integral",
             static_cast<AddGhostPenalty2d>(&Form2d::add_skeleton_integral),
             py::arg("penalty"), py::arg("quadrature"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("assemble", &Form2d::assemble, py::arg("assembler"));
}

} // namespace iguana::bindings
