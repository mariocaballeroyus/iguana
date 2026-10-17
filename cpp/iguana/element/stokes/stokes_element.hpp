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
 *        stabilized so that the velocity and the pressure take functions of
 *        one basis
 *
 * @details Its unknowns are the n components of the velocity and the
 *          pressure, in that order, the block size n + 1 of its space. The
 *          viscous stress is that of a Newtonian fluid, twice the viscosity
 *          times the symmetric gradient of the velocity. A variational
 *          multiscale method models the fine scales by the residuals: the
 *          gradients of the pressure test functions test the residual of the
 *          momentum equation, and a grad-div term that of the continuity
 *          equation. Its source is the body force, of n components, its
 *          traces the components of the velocity, U, and the pressure, P,
 *          and its fluxes the components of the traction, T
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class StokesElement final : public Element<Basis, n>
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
     *     τ_M = h² / (c μ)
     *     τ_C = μ
     *
     * @details The stabilization parameters of the momentum and continuity
     *          equations, with h the size of the element at each point
     *
     * @param viscosity Dynamic viscosity μ of the fluid
     * @param stabilization Constant c of τ_M, the larger the less it
     *        stabilizes
     *
     * @throws std::invalid_argument If the viscosity or the stabilization is
     *         not positive
     */
    StokesElement(Scalar viscosity, Scalar stabilization);

    /// @brief Dynamic viscosity μ of the fluid
    constexpr Scalar viscosity() const noexcept
    { return viscosity_; }

    /// @brief Constant c of the stabilization parameters
    constexpr Scalar stabilization() const noexcept
    { return stabilization_; }

    ValueFlags flags() const noexcept override
    { return {.physical_gradients = true, .physical_hessians = true}; }

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

private:
    /// @brief Stabilization parameter τ_M of the momentum equation at each
    ///        point of some values
    Eigen::VectorX<Scalar> tau(const ElementValues<Basis, n>& values) const;

    /// @brief Dynamic viscosity μ of the fluid
    Scalar viscosity_;

    /// @brief Constant c of the stabilization parameters
    Scalar stabilization_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_STOKES_STOKES_ELEMENT_HPP
