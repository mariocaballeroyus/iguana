/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_BINDINGS_PETSC_CASTER_HPP
#define IGUANA_BINDINGS_PETSC_CASTER_HPP

#include <petsc4py/petsc4py.h>
#include <pybind11/pybind11.h>

namespace pybind11::detail
{

/**
 * @brief Imports the C API of petsc4py into this translation unit, whose
 *        function pointers petsc4py.h keeps static, on first use
 *
 * @throws pybind11::error_already_set If petsc4py cannot be imported
 */
inline void import_petsc4py_once()
{
    if (PyPetscMat_New == nullptr && import_petsc4py() != 0)
        throw error_already_set();
}

/**
 * @brief Converts a PETSc matrix to and from a petsc4py.PETSc.Mat
 *
 * @details Both sides share one object, its references counted by PETSc,
 *          so nothing is copied and either side may outlive the other
 */
template<>
struct type_caster<_p_Mat>
{
    PYBIND11_TYPE_CASTER(Mat, const_name("petsc4py.PETSc.Mat"));

    bool load(handle source, bool)
    {
        import_petsc4py_once();

        if (PyObject_TypeCheck(source.ptr(), &PyPetscMat_Type) == 0)
            return false;

        value = PyPetscMat_Get(source.ptr());

        return true;
    }

    static handle cast(Mat source, return_value_policy, handle)
    {
        import_petsc4py_once();

        return PyPetscMat_New(source);
    }

    operator Mat()
    { return value; }
};

/**
 * @brief Converts a PETSc vector to and from a petsc4py.PETSc.Vec
 *
 * @details Both sides share one object, its references counted by PETSc,
 *          so nothing is copied and either side may outlive the other
 */
template<>
struct type_caster<_p_Vec>
{
    PYBIND11_TYPE_CASTER(Vec, const_name("petsc4py.PETSc.Vec"));

    bool load(handle source, bool)
    {
        import_petsc4py_once();

        if (PyObject_TypeCheck(source.ptr(), &PyPetscVec_Type) == 0)
            return false;

        value = PyPetscVec_Get(source.ptr());

        return true;
    }

    static handle cast(Vec source, return_value_policy, handle)
    {
        import_petsc4py_once();

        return PyPetscVec_New(source);
    }

    operator Vec()
    { return value; }
};

} // namespace pybind11::detail

#endif // IGUANA_BINDINGS_PETSC_CASTER_HPP
