/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_STOKES_STOKES_STABILIZED_ELEMENT_HPP
#define IGUANA_ELEMENT_STOKES_STOKES_STABILIZED_ELEMENT_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/stokes/stokes_element.hpp"

namespace iguana
{

/**
 * @brief Element of the Stokes problem of an incompressible viscous fluid,
 *        stabilized so that the velocity and the pressure take functions of
 *        one basis
 *
 * @details It adds to the Galerkin forms of StokesElement those of a
 *          variational multiscale method, which models the fine scales by
 *          the residuals: the gradients of the pressure test functions test
 *          the residual of the momentum equation, and a grad-div term that
 *          of the continuity equation. Its unknowns, source, traces and
 *          fluxes are those of StokesElement
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class StokesStabilizedElement final : public StokesElement<Basis, n>
{
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
    StokesStabilizedElement(Scalar viscosity, Scalar stabilization);

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

private:
    /// @brief Galerkin element, whose forms the stabilization adds to
    using Base = StokesElement<Basis, n>;

    /// @brief Stabilization parameter τ_M of the momentum equation at each
    ///        point of some values
    Eigen::VectorX<Scalar> tau(const ElementValues<Basis, n>& values) const;

    /// @brief Constant c of the stabilization parameters
    Scalar stabilization_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_STOKES_STOKES_STABILIZED_ELEMENT_HPP
