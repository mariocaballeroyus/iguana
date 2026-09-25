/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_DOMAIN_TENSOR_DOMAIN_ITERATOR_HPP
#define IGUANA_DOMAIN_TENSOR_DOMAIN_ITERATOR_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>

#include "iguana/basis/tensor_bspline.hpp"

namespace iguana
{

/**
 * @brief Walks the elements of a tensor domain
 *
 * The iterator visits every element once, in increasing flat index with
 * the first direction running fastest, and serves as the handle of the
 * element it has reached. Its data is refilled in place as it advances
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 *
 * @warning The iterator reads the basis of its domain, which must outlive
 *          it
 */
template<std::floating_point T, std::size_t d>
class TensorDomainIterator
{
public:
    /**
     * @brief Starts at the first element of a basis
     *
     * @param basis Basis whose elements are walked
     */
    explicit TensorDomainIterator(const TensorBSpline<T, d>& basis) noexcept;

    /// @brief Element handle, the iterator itself
    constexpr const TensorDomainIterator& operator*() const noexcept
    { return *this; }

    /// @brief Advances to the next element
    TensorDomainIterator& operator++() noexcept;

    /// @brief Whether the iterator has passed the last element
    constexpr bool operator==(std::default_sentinel_t) const noexcept
    { return index_ >= num_elements_; }

    /// @brief Flat element index, with the first direction running fastest
    constexpr int index() const noexcept
    { return index_; }

    /// @brief First active function of each direction
    constexpr const std::array<int, d>& first_active() const noexcept
    { return first_active_; }

    /// @brief Parameters at which the element starts
    constexpr const std::array<T, d>& start() const noexcept
    { return start_; }

    /// @brief Parameters at which the element ends
    constexpr const std::array<T, d>& end() const noexcept
    { return end_; }

private:
    /// @brief Refills the element data from the element of each direction
    void update() noexcept;

    /// @brief Basis whose elements are walked
    const TensorBSpline<T, d>* basis_ = nullptr;

    /// @brief Number of elements of the basis
    int num_elements_ = 0;

    /// @brief Number of elements of each direction
    std::array<int, d> element_counts_{};

    /// @brief Element of each direction
    std::array<int, d> axis_elements_{};

    /// @brief Flat element index
    int index_ = 0;

    /// @brief First active function of each direction
    std::array<int, d> first_active_{};

    /// @brief Parameters at which the element starts
    std::array<T, d> start_{};

    /// @brief Parameters at which the element ends
    std::array<T, d> end_{};
};

} // namespace iguana

#endif // IGUANA_DOMAIN_TENSOR_DOMAIN_ITERATOR_HPP
