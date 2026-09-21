/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>

#include "iguana/iguana.hpp"

PYBIND11_MODULE(iguana, module)
{
    module.doc() = "IGUANA: Immersogeometric framework for "
                   "boundary-unfitted analysis";
    module.attr("__version__") = IGUANA_VERSION;
}

