/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_STOKES_STOKES_ELEMENT_HPP
#define IGUANA_ELEMENT_STOKES_STOKES_ELEMENT_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/element.hpp"

namespace iguana
{

/**
 * @brief Element of the Stokes problem of an incompressible viscous fluid,
 *        in its Galerkin form
 *
 * @details Its unknowns are the n components of the velocity and the
 *          pressure, in that order, either the components of one field, the
 *          block size n + 1 of its space, or a field each, as in
 *          divergence-conforming spaces. The viscous stress is that of a
 *          Newtonian fluid, twice the viscosity times the symmetric gradient
 *          of the velocity. Without stabilization, the spaces of the
 *          velocity and the pressure must be inf-sup stable, while
 *          StokesStabilizedElement lets them take functions of one basis.
 *          Its source is the body force, of n components, its traces the
 *          components of the velocity, U, and the pressure, P, and its
 *          fluxes the components of the traction, T
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class StokesElement : public Element<Basis, n>
{
    static_assert(Basis::dimension == n,
                  "StokesElement: "
                  "the patch must be a domain");

public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the element of a fluid
     *
     * @param viscosity Dynamic viscosity μ of the fluid
     *
     * @throws std::invalid_argument If the viscosity is not positive
     */
    explicit StokesElement(Scalar viscosity);

    /// @brief Dynamic viscosity μ of the fluid
    constexpr Scalar viscosity() const noexcept
    { return viscosity_; }

    ValueFlags flags() const noexcept override
    { return {.physical_gradients = true}; }

    void local_stiffness(const ElementValues<Basis, n>& values,
                         const Eigen::VectorX<Scalar>& weights,
                         Eigen::MatrixX<Scalar>& stiffness) const override;

    void local_load(const ElementValues<Basis, n>& values,
                    const Eigen::VectorX<Scalar>& weights,
                    const Eigen::MatrixX<Scalar>& source,
                    Eigen::VectorX<Scalar>& load) const override;

    void local_mass(const ElementValues<Basis, n>& values,
                    const Eigen::VectorX<Scalar>& weights,
                    Eigen::MatrixX<Scalar>& mass) const override;

    /// @brief Trace of a component of the velocity, defined in
    ///        stokes_traces.hpp
    class U;

    /// @brief Trace of the pressure, defined in stokes_traces.hpp
    class P;

    /// @brief Flux of a component of the velocity, that component of the
    ///        traction, defined in stokes_traces.hpp
    class T;

protected:
    /**
     * @brief Field whose values an unknown reads, the components of the
     *        velocity followed by the pressure
     *
     * The pressure reads the last field, and each component of the velocity
     * its own of n + 1 fields, or else the first, which holds them as its
     * components
     *
     * @param num_fields Number of fields of the values
     * @param unknown Unknown, below n + 1
     */
    static constexpr std::size_t field_of(std::size_t num_fields,
                                          std::size_t unknown) noexcept
    {
        if (unknown == n)
            return num_fields - 1;

        return num_fields == n + 1 ? unknown : 0;
    }

    /**
     * @brief First row of an unknown in the local matrices, after the
     *        active functions of the unknowns before it
     *
     * @param values Values at the points of the element
     * @param unknown Unknown, up to n + 1, which gives the number of rows
     */
    static Eigen::Index first_row(const ElementValues<Basis, n>& values,
                                  std::size_t unknown)
    {
        Eigen::Index row = 0;

        for (std::size_t before = 0; before < unknown; ++before)
            row += values.values(field_of(values.num_fields(), before)).rows();

        return row;
    }

private:
    /// @brief Dynamic viscosity μ of the fluid
    Scalar viscosity_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_STOKES_STOKES_ELEMENT_HPP
