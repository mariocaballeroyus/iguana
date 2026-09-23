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

} // namespace iguana::bindings

PYBIND11_MODULE(cpp, module)
{
    module.attr("__version__") = IGUANA_VERSION;

    iguana::bindings::basis(module);
    iguana::bindings::patch(module);
}
