/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <stdexcept>

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <petscsys.h>

namespace
{

/**
 * @brief Initializes PETSc before the first test and finalizes it after the
 *        last, as the assemblers build their systems with it
 */
class PetscListener final : public Catch::EventListenerBase
{
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo&) override
    {
        if (PetscInitializeNoArguments() != PETSC_SUCCESS)
            throw std::runtime_error("PetscListener: "
                                     "PETSc failed to initialize");
    }

    void testRunEnded(const Catch::TestRunStats&) override
    {
        static_cast<void>(PetscFinalize());
    }
};

} // namespace

CATCH_REGISTER_LISTENER(PetscListener)
