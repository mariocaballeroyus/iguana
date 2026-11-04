/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_ELASTICITY_LINEAR_ELASTICITY_ELEMENT_HPP
#define IGUANA_ELEMENT_ELASTICITY_LINEAR_ELASTICITY_ELEMENT_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/element.hpp"

namespace iguana
{

/**
 * @brief Element of the linear elasticity problem of an isotropic solid under
 *        small displacements
 *
 *     −div σ(u) = f
 *     σ(u) = 2μ ε(u) + λ (div u) I
 *
 * @details Its unknowns are the n components of the displacement, those of
 *          one field, the block size n of its space. Its source is the body
 *          force, of n components, its traces the components of the
 *          displacement, U, and its fluxes those of the traction, T. In two
 *          dimensions the Lamé parameters state plane strain, and those of
 *          plane stress take λ = 2 λ μ / (λ + 2 μ). Its mass is that of unit
 *          density, the matrix of a first derivative in time, as the second
 *          one of elastodynamics has no time scheme yet
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class LinearElasticityElement : public Element<Basis, n>
{
    static_assert(Basis::dimension == n,
                  "LinearElasticityElement: "
                  "the patch must be a domain");

public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /**
     * @brief Constructs the element of a solid
     *
     * @param lame First Lamé parameter λ
     * @param shear_modulus Shear modulus μ, the second Lamé parameter
     *
     * @throws std::invalid_argument If the shear modulus is not positive, or
     *         if the bulk modulus λ + 2 μ / n is not, as the energy of every
     *         strain needs them to be
     */
    LinearElasticityElement(Scalar lame, Scalar shear_modulus);

    /// @brief First Lamé parameter λ
    constexpr Scalar lame() const noexcept
    { return lame_; }

    /// @brief Shear modulus μ
    constexpr Scalar shear_modulus() const noexcept
    { return shear_modulus_; }

    ValueFlags flags() const noexcept override
    { return {.physical_gradients = true}; }

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

    void local_mass(const ElementValues<Basis, n>& values,
                    const Eigen::VectorX<Scalar>& weights,
                    Eigen::MatrixX<Scalar>& mass) const override;

private:
    /// @brief First Lamé parameter λ
    Scalar lame_;

    /// @brief Shear modulus μ
    Scalar shear_modulus_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_ELASTICITY_LINEAR_ELASTICITY_ELEMENT_HPP
