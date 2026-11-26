/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_NAVIER_STOKES_NAVIER_STOKES_ELEMENT_HPP
#define IGUANA_ELEMENT_NAVIER_STOKES_NAVIER_STOKES_ELEMENT_HPP

#include <array>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/stokes/stokes_element.hpp"

namespace iguana
{

/**
 * @brief Element of the steady Navier-Stokes problem of an incompressible
 *        viscous fluid of unit density, its convection stabilized by a
 *        streamline-upwind term
 *
 *     ( 2μ ε(u), ε(v) ) + ( (u · ∇)u, v ) − ( p, div v ) − ( q, div u )
 *         + ( τ_M (u · ∇)v, r_M ) = ( f, v )
 *     r_M = (u · ∇)u − div(2μ ε(u)) + ∇p − f
 *
 * @details It adds to the Galerkin forms of StokesElement the convection of
 *          the velocity by itself, and a streamline-upwind term in which the
 *          test functions convected by the velocity test the residual r_M
 *          of the momentum equation. The spaces of the velocity and the
 *          pressure must be inf-sup stable, as those of a
 *          divergence-conforming space, while NavierStokesStabilizedElement
 *          lets them take functions of one basis. The residual is not affine
 *          in the coefficients, and the tangent is its exact derivative,
 *          that of τ_M with the velocity included, for Newton's method. Its
 *          unknowns, source, traces and fluxes are those of StokesElement
 *
 * @tparam Basis Basis of the patch, TensorBSpline, TensorNURBS or
 *         THBSpline
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class NavierStokesElement : public StokesElement<Basis, n>
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the element of a fluid
     *
     *     τ_M = 1 / (c μ / h² + 2 |u| / h)
     *
     * @details The stabilization parameter of the momentum equation, with h
     *          the size of the element and |u| the speed of the fluid at
     *          each point, that of StokesStabilizedElement at rest
     *
     * @param viscosity Dynamic viscosity μ of the fluid
     * @param stabilization Constant c of τ_M, the larger the less it
     *        stabilizes
     *
     * @throws std::invalid_argument If the viscosity or the stabilization is
     *         not positive
     */
    NavierStokesElement(Scalar viscosity, Scalar stabilization);

    /// @brief Constant c of the stabilization parameter τ_M
    constexpr Scalar stabilization() const noexcept
    { return stabilization_; }

    ValueFlags flags() const noexcept override
    { return {.physical_gradients = true, .physical_hessians = true}; }

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

protected:
    /// @brief The flow at the points of an element, each a value per point
    struct Flow
    {
        /// @brief Components u_a of the velocity
        std::array<Eigen::ArrayX<Scalar>, n> velocity;

        /// @brief Derivatives of the velocity, ∂_k u_a in gradient[a][k]
        std::array<std::array<Eigen::ArrayX<Scalar>, n>, n> gradient;

        /// @brief Components r_a of the residual of the momentum equation
        std::array<Eigen::ArrayX<Scalar>, n> momentum;

        /// @brief Speed |u|, kept off zero so that τ_M differentiates at
        ///        rest
        Eigen::ArrayX<Scalar> speed;

        /// @brief Stabilization parameter τ_M of the momentum equation
        Eigen::ArrayX<Scalar> tau;
    };

    /**
     * @brief Flow at the points of an element for some coefficients
     *
     * @param values Values at the points, with their physical gradients and
     *        Hessians
     * @param source Body force at the points, a row each
     * @param coefficients Local coefficients of the unknowns
     */
    Flow flow(const ElementValues<Basis, n>& values,
              const Eigen::MatrixX<Scalar>& source,
              const Eigen::VectorX<Scalar>& coefficients) const;

    /**
     * @brief Functions convected by the velocity, u · ∇N_i at each point
     *
     * @param grads Physical gradients of the functions, one matrix per
     *        direction with a row per function and a column per point
     * @param state Flow at the points
     * @return A row per function and a column per point
     */
    static Eigen::MatrixX<Scalar> convected(
        const std::array<Eigen::MatrixX<Scalar>, n>& grads, const Flow& state);

    /**
     * @brief Derivatives of a component of the residual of the momentum
     *        equation by the coefficients of a component of the velocity, at
     *        each point
     *
     *     ∂r_a / ∂c_j = N_j ∂_b u_a + δ_ab u · ∇N_j
     *                   − μ (∂_a ∂_b N_j + δ_ab ΔN_j)
     *
     * @details With c_j the coefficient of N_j e_b
     *
     * @param values Values at the points, with their physical gradients and
     *        Hessians
     * @param state Flow at the points
     * @param a Component of the residual
     * @param b Component of the velocity
     * @return A row per function of component b and a column per point
     */
    Eigen::MatrixX<Scalar> momentum_derivative(
        const ElementValues<Basis, n>& values, const Flow& state,
        std::size_t a, std::size_t b) const;

private:
    /// @brief Galerkin element, whose forms the convection adds to
    using Base = StokesElement<Basis, n>;

    /// @brief Constant c of the stabilization parameter τ_M
    Scalar stabilization_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_NAVIER_STOKES_NAVIER_STOKES_ELEMENT_HPP
