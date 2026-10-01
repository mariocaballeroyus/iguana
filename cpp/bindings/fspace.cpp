/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <span>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using FunctionSpace2d = FunctionSpace<TensorBSpline<double, 2>>;
using FunctionSpace3d = FunctionSpace<TensorBSpline<double, 3>>;

/// @brief Number of degrees of freedom, read from the map of the space
template<std::size_t d>
int num_dofs(const FunctionSpace<TensorBSpline<double, d>>& space)
{
    return space.dof_map().num_dofs();
}

/// @brief Basis function of each degree of freedom, which ties it to a
///        control point of the patch
template<std::size_t d>
Eigen::VectorXi functions(
    const FunctionSpace<TensorBSpline<double, d>>& space)
{
    const DofMap& dof_map = space.dof_map();
    Eigen::VectorXi result(dof_map.num_dofs());
    Eigen::VectorXi actives;

    // The k-th degree of freedom of a covered cell belongs to its k-th
    // active function
    for (int element = 0; element < dof_map.num_elements(); ++element) {
        const std::span<const int> dofs = dof_map.dofs_on_element(element);

        if (dofs.empty())
            continue;

        space.basis().active_on_element(element, actives);
        result(dofs) = actives;
    }

    return result;
}

} // namespace

void fspace(py::module_& module)
{
    py::class_<FunctionSpace2d>(module, "FunctionSpace2d")
        .def(py::init<TensorBSpline<double, 2>, const Embedding<double, 2>&>(),
             py::arg("basis"), py::arg("embedding"))
        .def_property_readonly("num_dofs", &num_dofs<2>)
        .def_property_readonly("functions", &functions<2>);

    py::class_<FunctionSpace3d>(module, "FunctionSpace3d")
        .def(py::init<TensorBSpline<double, 3>, const Embedding<double, 3>&>(),
             py::arg("basis"), py::arg("embedding"))
        .def_property_readonly("num_dofs", &num_dofs<3>)
        .def_property_readonly("functions", &functions<3>);
}

} // namespace iguana::bindings
