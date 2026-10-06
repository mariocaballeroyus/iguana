/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_POISSON_POISSON_ELEMENT_HPP
#define IGUANA_ELEMENT_POISSON_POISSON_ELEMENT_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/element.hpp"

namespace iguana
{

/**
 * @brief Element of the Poisson problem -div(grad u) = f
 *
 * Its stiffness pairs the physical gradients of the active functions,
 * K_AB = integral of grad N_A . grad N_B, and its load weights their values
 * by the source, F_A = integral of f N_A. It holds no state. The patch must
 * be a domain, as only a domain has physical gradients. Its only trace is
 * the field, U, the flux being natural
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class PoissonElement final : public Element<Basis, n>
{
    static_assert(Basis::dimension == n,
                  "PoissonElement: "
                  "the patch must be a domain");

public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    ValueFlags flags() const noexcept override
    { return {.physical_gradients = true}; }

    void local_stiffness(const ElementValues<Basis, n>& values,
                         const Eigen::VectorX<Scalar>& weights,
                         Eigen::MatrixX<Scalar>& stiffness) const override;

    void local_load(const ElementValues<Basis, n>& values,
                    const Eigen::VectorX<Scalar>& weights,
                    const Eigen::VectorX<Scalar>& source,
                    Eigen::VectorX<Scalar>& load) const override;

    /// @brief Trace of the field u, defined in poisson_traces.hpp
    class U;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_POISSON_POISSON_ELEMENT_HPP
