/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using Trace2d = Trace<TensorBSpline<double, 2>, 2>;
using Flux2d = Flux<TensorBSpline<double, 2>, 2>;
using PoissonElement2d = PoissonElement<TensorBSpline<double, 2>, 2>;
using PoissonElement3d = PoissonElement<TensorBSpline<double, 3>, 3>;
using PenaltyCondition2d = PenaltyCondition<TensorBSpline<double, 2>, 2>;
using NeumannCondition2d = NeumannCondition<TensorBSpline<double, 2>, 2>;
using NitscheCondition2d = NitscheCondition<TensorBSpline<double, 2>, 2>;
using Assembler2d = Assembler<TensorBSpline<double, 2>, 2>;
using Assembler3d = Assembler<TensorBSpline<double, 3>, 3>;

} // namespace

void assembly(py::module_& module)
{
    // The system is copied out, the stiffness as a sparse matrix, as each
    // assembly adds into it
    constexpr py::return_value_policy copy = py::return_value_policy::copy;

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

    py::class_<PoissonElement3d>(module, "PoissonElement3d")
        .def(py::init<>());

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

    // An assembler keeps references to its space and patch, which Python
    // keeps alive with it. The wrapper of the solve drives it
    py::class_<Assembler2d>(module, "Assembler2d")
        .def(py::init<const FunctionSpace<TensorBSpline<double, 2>>&,
                      const Patch<TensorBSpline<double, 2>, 2>&>(),
             py::arg("space"), py::arg("patch"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def("assemble_stiffness",
             &Assembler2d::assemble_stiffness<PoissonElement2d>,
             py::arg("element"), py::arg("quadrature"))
        .def("assemble_load", &Assembler2d::assemble_load<PoissonElement2d>,
             py::arg("element"), py::arg("quadrature"), py::arg("source"))
        .def("assemble_stiffness",
             &Assembler2d::assemble_stiffness<PenaltyCondition2d>,
             py::arg("condition"), py::arg("quadrature"))
        .def("assemble_load",
             &Assembler2d::assemble_load<PenaltyCondition2d>,
             py::arg("condition"), py::arg("quadrature"), py::arg("data"))
        .def("assemble_load",
             &Assembler2d::assemble_load<NeumannCondition2d>,
             py::arg("condition"), py::arg("quadrature"), py::arg("data"))
        .def("assemble_stiffness",
             &Assembler2d::assemble_stiffness<NitscheCondition2d>,
             py::arg("condition"), py::arg("quadrature"))
        .def("assemble_load",
             &Assembler2d::assemble_load<NitscheCondition2d>,
             py::arg("condition"), py::arg("quadrature"), py::arg("data"))
        .def_property_readonly("stiffness", &Assembler2d::stiffness, copy)
        .def_property_readonly("load", &Assembler2d::load, copy);

    py::class_<Assembler3d>(module, "Assembler3d")
        .def(py::init<const FunctionSpace<TensorBSpline<double, 3>>&,
                      const Patch<TensorBSpline<double, 3>, 3>&>(),
             py::arg("space"), py::arg("patch"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def("assemble_stiffness",
             &Assembler3d::assemble_stiffness<PoissonElement3d>,
             py::arg("element"), py::arg("quadrature"))
        .def("assemble_load", &Assembler3d::assemble_load<PoissonElement3d>,
             py::arg("element"), py::arg("quadrature"), py::arg("source"))
        .def_property_readonly("stiffness", &Assembler3d::stiffness, copy)
        .def_property_readonly("load", &Assembler3d::load, copy);
}

} // namespace iguana::bindings
