/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_POISSON_POISSON_TRACES_HPP
#define IGUANA_ELEMENT_POISSON_POISSON_TRACES_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/flux.hpp"
#include "iguana/element/poisson/poisson_element.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Trace of the field u of the Poisson problem, its only essential
 *        quantity, the flux being natural
 *
 * Its matrix B holds the values of the active functions at the boundary
 * points
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class PoissonElement<Basis, n>::U final : public Trace<Basis, n>
{
public:
    ValueFlags flags() const noexcept override
    { return {}; }

    void local_trace(const ElementValues<Basis, n>& values,
                     Eigen::MatrixX<Scalar>& trace) const override
    { trace = values.values(); }
};

/**
 * @brief Flux of the field u of the Poisson problem, its normal derivative
 *        du/dn, conjugate to the trace U
 *
 * Its matrix F holds the physical gradients of the active functions along
 * the outward normals at the boundary points: the derivative itself,
 * without the sign or the conductivity of a heat flux
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class PoissonElement<Basis, n>::Q final : public Flux<Basis, n>
{
public:
    ValueFlags flags() const noexcept override
    { return {.physical_gradients = true}; }

    void local_flux(const ElementValues<Basis, n>& values,
                    const PointMatrix<Scalar, n>& normals,
                    Eigen::MatrixX<Scalar>& flux) const override
    {
        flux.setZero(values.values().rows(), values.values().cols());

        // F_iq = ∇N_i(x_q) · n(x_q)
        for (std::size_t coordinate = 0; coordinate < n; ++coordinate)
            flux += values.physical_gradients()[coordinate]
                    * normals.col(coordinate).asDiagonal();
    }
};

} // namespace iguana

#endif // IGUANA_ELEMENT_POISSON_POISSON_TRACES_HPP
