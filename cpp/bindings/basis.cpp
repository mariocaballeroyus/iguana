/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

/// @brief Univariate basis of a direction, from the B-splines that the
///        weights rationalize
template<std::size_t d>
const BSpline<double>& nurbs_axis(const TensorNURBS<double, d>& basis,
                                  std::size_t direction)
{
    return basis.bspline().axis(direction);
}

} // namespace

void basis(py::module_& module)
{
    py::class_<BSpline<double>>(module, "BSpline")
        .def(py::init<int, std::vector<double>>(),
             py::arg("degree"), py::arg("knots"))
        .def_property_readonly("degree", &BSpline<double>::degree)
        .def_property_readonly("knots", [](const BSpline<double>& basis) {
            return basis.knots().values();
        });

    py::class_<TensorBSpline<double, 1>>(module, "UnivariateBSpline")
        .def(py::init<std::array<BSpline<double>, 1>>(), py::arg("axes"))
        .def("axis", &TensorBSpline<double, 1>::axis);

    py::class_<TensorBSpline<double, 2>>(module, "BivariateBSpline")
        .def(py::init<std::array<BSpline<double>, 2>>(), py::arg("axes"))
        .def("axis", &TensorBSpline<double, 2>::axis);

    py::class_<TensorBSpline<double, 3>>(module, "TrivariateBSpline")
        .def(py::init<std::array<BSpline<double>, 3>>(), py::arg("axes"))
        .def("axis", &TensorBSpline<double, 3>::axis);

    py::class_<TensorNURBS<double, 1>>(module, "UnivariateNURBS")
        .def(py::init<TensorBSpline<double, 1>, Eigen::VectorXd>(),
             py::arg("bspline"), py::arg("weights"))
        .def("axis", &nurbs_axis<1>)
        .def_property_readonly("weights", &TensorNURBS<double, 1>::weights);

    py::class_<TensorNURBS<double, 2>>(module, "BivariateNURBS")
        .def(py::init<TensorBSpline<double, 2>, Eigen::VectorXd>(),
             py::arg("bspline"), py::arg("weights"))
        .def("axis", &nurbs_axis<2>)
        .def_property_readonly("weights", &TensorNURBS<double, 2>::weights);
}

} // namespace iguana::bindings
