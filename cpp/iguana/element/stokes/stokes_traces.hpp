/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_STOKES_STOKES_TRACES_HPP
#define IGUANA_ELEMENT_STOKES_STOKES_TRACES_HPP

#include <array>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/flux.hpp"
#include "iguana/element/stokes/stokes_element.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Trace of a component of the velocity of the Stokes problem, on
 *        which a condition imposes values such as no slip
 *
 * @details Its matrix B holds the values of the active functions at the
 *          boundary points in the rows of its component, and zeros in those
 *          of the other components and of the pressure
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class StokesElement<Basis, n>::U final : public Trace<Basis, n>
{
public:
    /**
     * @brief Constructs the trace of a component of the velocity
     *
     * @param component Component of the velocity, below n
     *
     * @throws std::invalid_argument If the component is not one of the
     *         velocity
     */
    explicit U(std::size_t component) : component_(component)
    {
        if (component >= n)
            throw std::invalid_argument("StokesElement::U: "
                                        "the component must be one of the "
                                        "velocity");
    }

    /// @brief Component of the velocity
    constexpr std::size_t component() const noexcept
    { return component_; }

    ValueFlags flags() const noexcept override
    { return {}; }

    std::size_t field(std::size_t) const noexcept override
    { return 0; }

    void local_trace(const ElementValues<Basis, n>& values,
                     Eigen::MatrixX<Scalar>& trace) const override
    {
        const Eigen::Index rows = values.values().rows();
        const Eigen::Index row = static_cast<Eigen::Index>(component_) * rows;

        // B_iq = N_i(x_q) in the rows of the component
        trace.setZero(static_cast<Eigen::Index>(n + 1) * rows,
                      values.values().cols());
        trace.middleRows(row, rows) = values.values();
    }

private:
    /// @brief Component of the velocity
    std::size_t component_;
};

/**
 * @brief Trace of the pressure of the Stokes problem, which a ghost penalty
 *        extends beyond the fluid
 *
 * @details Its matrix B holds the values of the active functions at the
 *          points in the rows of the pressure, and zeros in those of the
 *          velocity
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class StokesElement<Basis, n>::P final : public Trace<Basis, n>
{
public:
    ValueFlags flags() const noexcept override
    { return {}; }

    std::size_t field(std::size_t) const noexcept override
    { return 0; }

    void local_trace(const ElementValues<Basis, n>& values,
                     Eigen::MatrixX<Scalar>& trace) const override
    {
        const Eigen::Index rows = values.values().rows();
        const Eigen::Index row = static_cast<Eigen::Index>(n) * rows;

        // B_iq = N_i(x_q) in the rows of the pressure
        trace.setZero(row + rows, values.values().cols());
        trace.middleRows(row, rows) = values.values();
    }
};

/**
 * @brief Flux of a component of the velocity of the Stokes problem, that
 *        component of the traction, conjugate to its trace U
 *
 * @details The traction is the stress of the fluid applied to the normal,
 *          the stress being twice the viscosity times the symmetric
 *          gradient of the velocity, less the pressure
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class StokesElement<Basis, n>::T final : public Flux<Basis, n>
{
public:
    /**
     * @brief Constructs the flux of a component of the velocity
     *
     * @param viscosity Dynamic viscosity μ of the fluid, that of the
     *        element
     * @param component Component of the traction, below n
     *
     * @throws std::invalid_argument If the viscosity is not positive, or if
     *         the component is not one of the velocity
     */
    T(Scalar viscosity, std::size_t component)
        : viscosity_(viscosity), component_(component)
    {
        if (viscosity <= 0)
            throw std::invalid_argument("StokesElement::T: "
                                        "the viscosity must be positive");

        if (component >= n)
            throw std::invalid_argument("StokesElement::T: "
                                        "the component must be one of the "
                                        "velocity");
    }

    /// @brief Dynamic viscosity μ of the fluid
    constexpr Scalar viscosity() const noexcept
    { return viscosity_; }

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
        const Eigen::Index rows = values.values().rows();
        const Eigen::Index pressure_row = static_cast<Eigen::Index>(n) * rows;

        // ∂N_i/∂n, the derivatives along the normal
        Eigen::MatrixX<Scalar> normal_derivatives =
            Eigen::MatrixX<Scalar>::Zero(rows, values.values().cols());

        for (std::size_t coordinate = 0; coordinate < n; ++coordinate)
            normal_derivatives +=
                grads[coordinate] * normals.col(coordinate).asDiagonal();

        flux.setZero(pressure_row + rows, values.values().cols());

        // F_iq = μ (δ_ab ∂N_i/∂n + ∂_a N_i n_b)(x_q) in the rows of the
        // component b of the velocity, for the component a of the traction
        for (std::size_t b = 0; b < n; ++b) {
            const Eigen::Index row = static_cast<Eigen::Index>(b) * rows;

            flux.middleRows(row, rows) =
                viscosity_ * grads[component_] * normals.col(b).asDiagonal();

            if (b == component_)
                flux.middleRows(row, rows) += viscosity_ * normal_derivatives;
        }

        // F_iq = −N_i(x_q) n_a(x_q) in the rows of the pressure
        flux.middleRows(pressure_row, rows) =
            -values.values() * normals.col(component_).asDiagonal();
    }

private:
    /// @brief Dynamic viscosity μ of the fluid
    Scalar viscosity_;

    /// @brief Component of the traction
    std::size_t component_;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_STOKES_STOKES_TRACES_HPP
