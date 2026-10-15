/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace iguana::bindings
{

void basis(py::module_& module);
void patch(py::module_& module);
void boundary(py::module_& module);
void grid(py::module_& module);
void embedding(py::module_& module);
void fspace(py::module_& module);
void quadrature(py::module_& module);

#ifdef IGUANA_WITH_PETSC
void assembly(py::module_& module);
#endif

} // namespace iguana::bindings

PYBIND11_MODULE(cpp, module)
{
    module.attr("__version__") = IGUANA_VERSION;

    iguana::bindings::basis(module);
    iguana::bindings::patch(module);
    iguana::bindings::boundary(module);
    iguana::bindings::grid(module);
    iguana::bindings::embedding(module);
    iguana::bindings::fspace(module);
    iguana::bindings::quadrature(module);

#ifdef IGUANA_WITH_PETSC
    iguana::bindings::assembly(module);
#endif
}
