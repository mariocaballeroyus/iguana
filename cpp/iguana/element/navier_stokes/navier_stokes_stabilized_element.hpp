/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_NAVIER_STOKES_NAVIER_STOKES_STABILIZED_ELEMENT_HPP
#define IGUANA_ELEMENT_NAVIER_STOKES_NAVIER_STOKES_STABILIZED_ELEMENT_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/navier_stokes/navier_stokes_element.hpp"

namespace iguana
{

/**
 * @brief Element of the steady Navier-Stokes problem of an incompressible
 *        viscous fluid of unit density, stabilized so that the velocity and
 *        the pressure take functions of one basis
 *
 *     ( 2μ ε(u), ε(v) ) + ( (u · ∇)u, v ) − ( p, div v ) − ( q, div u )
 *         + ( τ_M (u · ∇)v − τ_M ∇q, r_M ) + ( τ_C div u, div v )
 *         = ( f, v )
 *     r_M = (u · ∇)u − div(2μ ε(u)) + ∇p − f
 *
 * @details It adds to the forms of NavierStokesElement those that
 *          StokesStabilizedElement adds to the Galerkin ones: the gradients
 *          of the pressure test functions test the residual r_M of the
 *          momentum equation, and a grad-div term that of the continuity
 *          equation. The tangent is the exact derivative of the residual,
 *          those of τ_M and τ_C with the velocity included. Its unknowns,
 *          source, traces and fluxes are those of StokesElement
 *
 * @tparam Basis Basis of the patch, TensorBSpline, TensorNURBS or
 *         THBSpline
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class NavierStokesStabilizedElement final
    : public NavierStokesElement<Basis, n>
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the element of a fluid
     *
     *     τ_M = 1 / (c μ / h² + 2 |u| / h)
     *     τ_C = h² / (c τ_M) = μ + 2 |u| h / c
     *
     * @details The stabilization parameters of the momentum and continuity
     *          equations, with h the size of the element and |u| the speed
     *          of the fluid at each point, those of StokesStabilizedElement
     *          at rest
     *
     * @param viscosity Dynamic viscosity μ of the fluid
     * @param stabilization Constant c of the stabilization parameters, the
     *        larger the less they stabilize
     *
     * @throws std::invalid_argument If the viscosity or the stabilization is
     *         not positive
     */
    NavierStokesStabilizedElement(Scalar viscosity, Scalar stabilization);

    void local_tangent(const ElementValues<Basis, n>& values,
                       const Eigen::VectorX<Scalar>& weights,
                       const Eigen::MatrixX<Scalar>& source,
                       const Eigen::VectorX<Scalar>& coefficients,
                       Eigen::MatrixX<Scalar>& tangent) const override;

    void local_residual(const ElementValues<Basis, n>& values,
                        const Eigen::VectorX<Scalar>& weights,
                        const Eigen::MatrixX<Scalar>& source,
                        const Eigen::VectorX<Scalar>& coefficients,
                        Eigen::VectorX<Scalar>& residual) const override;

private:
    /// @brief Element whose forms the stabilization adds to
    using Base = NavierStokesElement<Basis, n>;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_NAVIER_STOKES_NAVIER_STOKES_STABILIZED_ELEMENT_HPP
