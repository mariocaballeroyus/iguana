/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
#define IGUANA_DOMAIN_TENSOR_DOMAIN_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "iguana/domain/knot_vector.hpp"

namespace iguana
{

/**
 * @brief Elements of a tensor-product basis, the products of the elements
 *        of one knot vector per direction
 *
 * The elements are numbered with the first direction running fastest
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
     * @brief Constructs the domain from the knot vector of each direction
     *
     * @param knots Knot vector of each parametric direction
     */
    explicit TensorDomain(std::array<KnotVector<T>, d> knots);

    /**
     * @brief Knot vector of a parametric direction
     *
     * @param direction Direction index
     *
     * @pre @p direction lies in [0, dimension)
     */
    constexpr const KnotVector<T>& knots(std::size_t direction) const noexcept
    { return knots_[direction]; }

    /// @brief Number of elements, the product of those of each direction
    constexpr int num_elements() const noexcept
    { return num_elements_; }

private:
    /// @brief Knot vector of each parametric direction
    std::array<KnotVector<T>, d> knots_;

    /// @brief Product of the element counts of each direction
    int num_elements_;
};

} // namespace iguana

#endif // IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
