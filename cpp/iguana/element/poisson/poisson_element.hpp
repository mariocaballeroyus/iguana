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
 * @brief Element of the Poisson problem −div(grad u) = f, and of the heat
 *        equation du/dt − div(grad u) = f with its mass
 *
 * @details It holds no state, and its patch must be a domain, the only one
 *          with physical gradients. Its source has one component, its trace
 *          is the field, U, and its flux the normal derivative of the
 *          field, Q
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

    /// @brief Trace of the field u, defined in poisson_traces.hpp
    class U;

    /// @brief Flux of the field u, its normal derivative, defined in
    ///        poisson_traces.hpp
    class Q;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_POISSON_POISSON_ELEMENT_HPP
