/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_ELASTICITY_ELASTICITY_ELEMENT_HPP
#define IGUANA_ELEMENT_ELASTICITY_ELASTICITY_ELEMENT_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/element.hpp"

namespace iguana
{

/**
 * @brief Element of an isotropic elastic solid, the St. Venant–Kirchhoff
 *        problem under large displacements and moderate strains
 *
 *     −div (F S) = f
 *     F = I + ∇u
 *     E = ½ (FᵀF − I)
 *     S = 2μ E + λ tr(E) I
 *
 * @details The patch maps the reference configuration, over which the
 *          gradients and integrals are taken, so that the source is the body
 *          force per unit of reference volume. At rest, its tangent is the
 *          stiffness of linear elasticity and its residual minus the load, so
 *          that a linear problem solves the theory of small displacements and
 *          a nonlinear problem that of large ones. Its unknowns are the n
 *          components of the displacement, those of one field, the block size
 *          n of its space. Its source is the body force, of n components, its
 *          traces the components of the displacement, U, and its fluxes those
 *          of the traction under small displacements, T. In two dimensions
 *          the Lamé parameters state plane strain, and those of plane stress
 *          take λ = 2 λ μ / (λ + 2 μ). Its mass is that of unit density, the
 *          matrix of a first derivative in time, as the second one of
 *          elastodynamics has no time scheme yet
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class ElasticityElement : public Element<Basis, n>
{
    static_assert(Basis::dimension == n,
                  "ElasticityElement: "
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
    ElasticityElement(Scalar lame, Scalar shear_modulus);

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

    /// @brief Trace of a component of the displacement, defined in
    ///        elasticity_traces.hpp
    class U;

    /// @brief Flux of a component of the displacement, that component of the
    ///        traction under small displacements, defined in
    ///        elasticity_traces.hpp
    class T;

private:
    /// @brief Second Piola–Kirchhoff stress S of a deformation gradient F
    Eigen::Matrix<Scalar, n, n> stress(
        const Eigen::Matrix<Scalar, n, n>& gradient) const;

    /// @brief First Lamé parameter λ
    Scalar lame_;

    /// @brief Shear modulus μ
    Scalar shear_modulus_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_ELASTICITY_ELASTICITY_ELEMENT_HPP
