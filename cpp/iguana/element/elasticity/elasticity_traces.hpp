/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_ELASTICITY_ELASTICITY_TRACES_HPP
#define IGUANA_ELEMENT_ELASTICITY_ELASTICITY_TRACES_HPP

#include <array>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/elasticity/elasticity_element.hpp"
#include "iguana/element/flux.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Trace of a component of the displacement of an elastic solid, on
 *        which a condition imposes values such as a clamped side
 *
 * @details Its matrix B holds the values of the active functions at the
 *          boundary points in the rows of its component, and zeros in those
 *          of the other components
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class ElasticityElement<Basis, n>::U final : public Trace<Basis, n>
{
public:
    /**
     * @brief Constructs the trace of a component of the displacement
     *
     * @param component Component of the displacement, below n
     *
     * @throws std::invalid_argument If the component is not one of the
     *         displacement
     */
    explicit U(std::size_t component) : component_(component)
    {
        if (component >= n)
            throw std::invalid_argument("ElasticityElement::U: "
                                        "the component must be one of the "
                                        "displacement");
    }

    /// @brief Component of the displacement
    constexpr std::size_t component() const noexcept
    { return component_; }

    ValueFlags flags() const noexcept override
    { return {}; }

    std::size_t field(std::size_t) const noexcept override
    { return 0; }

    void local_trace(const ElementValues<Basis, n>& values,
                     Eigen::MatrixX<Scalar>& trace) const override
    {
        const Eigen::MatrixX<Scalar>& functions = values.values();
        const Eigen::Index rows = functions.rows();

        // B_iq = N_i(x_q) in the rows of the component
        trace.setZero(static_cast<Eigen::Index>(n) * rows, functions.cols());
        trace.middleRows(static_cast<Eigen::Index>(component_) * rows, rows) =
            functions;
    }

private:
    /// @brief Component of the displacement
    std::size_t component_;
};

/**
 * @brief Flux of a component of the displacement of an elastic solid, that
 *        component of the traction under small displacements, conjugate to
 *        its trace U
 *
 *     t = σ(u) n = (2μ ε(u) + λ (div u) I) n
 *
 * @details The traction is the stress of the solid applied to the normal. A
 *          flux is linear in the coefficients, so this is the traction of the
 *          element linearized at rest, which a linear problem solves, and not
 *          that under large displacements
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class ElasticityElement<Basis, n>::T final : public Flux<Basis, n>
{
public:
    /**
     * @brief Constructs the flux of a component of the displacement
     *
     * @param lame First Lamé parameter λ, that of the element
     * @param shear_modulus Shear modulus μ, that of the element
     * @param component Component of the traction, below n
     *
     * @throws std::invalid_argument If the shear modulus or the bulk modulus
     *         is not positive, or if the component is not one of the
     *         displacement
     */
    T(Scalar lame, Scalar shear_modulus, std::size_t component)
        : lame_(lame), shear_modulus_(shear_modulus), component_(component)
    {
        if (shear_modulus <= 0)
            throw std::invalid_argument("ElasticityElement::T: "
                                        "the shear modulus must be positive");

        if (lame + 2 * shear_modulus / static_cast<Scalar>(n) <= 0)
            throw std::invalid_argument("ElasticityElement::T: "
                                        "the bulk modulus must be positive");

        if (component >= n)
            throw std::invalid_argument("ElasticityElement::T: "
                                        "the component must be one of the "
                                        "displacement");
    }

    /// @brief First Lamé parameter λ
    constexpr Scalar lame() const noexcept
    { return lame_; }

    /// @brief Shear modulus μ
    constexpr Scalar shear_modulus() const noexcept
    { return shear_modulus_; }

    /// @brief Component of the traction
    constexpr std::size_t component() const noexcept
    { return component_; }

    ValueFlags flags() const noexcept override
    { return {.physical_gradients = true}; }

    void local_flux(const ElementValues<Basis, n>& values,
                    const PointMatrix<Scalar, n>& normals,
                    Eigen::MatrixX<Scalar>& flux) const override
    {
        const std::array<Eigen::MatrixX<Scalar>, n>& grads =
            values.physical_gradients();
        const Eigen::Index rows = grads[0].rows();

        flux.resize(static_cast<Eigen::Index>(n) * rows, grads[0].cols());

        // F_iq = λ ∂_b N_i n_a + μ (δ_ab ∂N_i/∂n + ∂_a N_i n_b) at x_q, in
        // the rows of the component b of the displacement, for the component
        // a of the traction
        for (std::size_t b = 0; b < n; ++b) {
            const Eigen::Index row = static_cast<Eigen::Index>(b) * rows;

            flux.middleRows(row, rows) =
                lame_ * grads[b] * normals.col(component_).asDiagonal()
                + shear_modulus_ * grads[component_]
                      * normals.col(b).asDiagonal();

            // ∂N_i/∂n, the derivatives along the normal
            if (b == component_) {
                for (std::size_t c = 0; c < n; ++c)
                    flux.middleRows(row, rows) +=
                        shear_modulus_ * grads[c]
                        * normals.col(c).asDiagonal();
            }
        }
    }

private:
    /// @brief First Lamé parameter λ
    Scalar lame_;

    /// @brief Shear modulus μ
    Scalar shear_modulus_;

    /// @brief Component of the traction
    std::size_t component_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_ELASTICITY_ELASTICITY_TRACES_HPP
