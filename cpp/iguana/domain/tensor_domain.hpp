/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
#define IGUANA_DOMAIN_TENSOR_DOMAIN_HPP

#include <concepts>
#include <cstddef>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/patch/patch.hpp"

namespace iguana
{

/**
 * @brief Domain over the elements of a tensor-product patch
 *
 * The domain is the physical domain discretized over the elements of a
 * background patch. For now every element belongs to it, until immersed
 * geometry classifies the elements it cuts
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class TensorDomain
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the domain over the elements of a patch
     *
     * @param patch Background patch, whose basis gives the elements
     */
    explicit TensorDomain(Patch<T, d> patch);

    /// @brief Background patch
    constexpr const Patch<T, d>& patch() const noexcept
    { return patch_; }

    /// @brief Basis of the background patch, whose elements the domain has
    constexpr const TensorBSpline<T, d>& basis() const noexcept
    { return patch_.basis(); }

    /// @brief Number of elements
    constexpr int num_elements() const noexcept
    { return patch_.basis().num_elements(); }

private:
    /// @brief Background patch
    Patch<T, d> patch_;
};

} // namespace iguana

#endif // IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
