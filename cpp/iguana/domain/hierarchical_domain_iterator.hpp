/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_DOMAIN_HIERARCHICAL_DOMAIN_ITERATOR_HPP
#define IGUANA_DOMAIN_HIERARCHICAL_DOMAIN_ITERATOR_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>

namespace iguana
{

template<std::floating_point T, std::size_t d>
class HierarchicalDomain;

/**
 * @brief Walks the active elements of a hierarchical domain
 *
 * The iterator visits every active element once, in increasing
 * hierarchical index, that is level by level and, within a level, in
 * increasing index of the level domain. It serves as the handle of the
 * element it has reached, and its data is refilled in place as it
 * advances
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 *
 * @warning The iterator reads its domain, which must outlive it
 */
template<std::floating_point T, std::size_t d>
class HierarchicalDomainIterator
{
public:
    /**
     * @brief Starts at the first active element of a domain
     *
     * @param domain Domain whose active elements are walked
     */
    explicit HierarchicalDomainIterator(
        const HierarchicalDomain<T, d>& domain) noexcept;

    /// @brief Element handle, the iterator itself
    constexpr const HierarchicalDomainIterator& operator*() const noexcept
    { return *this; }

    /// @brief Advances to the next active element
    HierarchicalDomainIterator& operator++() noexcept;

    /// @brief Whether the iterator has passed the last active element
    constexpr bool operator==(std::default_sentinel_t) const noexcept
    { return index_ >= num_elements_; }

    /// @brief Hierarchical element index
    constexpr int index() const noexcept
    { return index_; }

    /// @brief Level of the element
    constexpr int level() const noexcept
    { return level_; }

    /// @brief First active function of each direction, in the tensor basis
    ///        of the level
    constexpr const std::array<int, d>& first_active() const noexcept
    { return first_active_; }

    /// @brief Parameters at which the element starts
    constexpr const std::array<T, d>& start() const noexcept
    { return start_; }

    /// @brief Parameters at which the element ends
    constexpr const std::array<T, d>& end() const noexcept
    { return end_; }

private:
    /// @brief Skips the levels whose active elements are exhausted, then
    ///        refills the element data unless the walk is over
    void settle() noexcept;

    /// @brief Refills the element data from the level domain
    void update() noexcept;

    /// @brief Domain whose active elements are walked
    const HierarchicalDomain<T, d>* domain_ = nullptr;

    /// @brief Number of active elements of the domain
    int num_elements_ = 0;

    /// @brief Hierarchical element index
    int index_ = 0;

    /// @brief Level of the element
    int level_ = 0;

    /// @brief Position of the element among the active ones of its level
    int position_ = 0;

    /// @brief First active function of each direction
    std::array<int, d> first_active_{};

    /// @brief Parameters at which the element starts
    std::array<T, d> start_{};

    /// @brief Parameters at which the element ends
    std::array<T, d> end_{};
};

} // namespace iguana

#endif // IGUANA_DOMAIN_HIERARCHICAL_DOMAIN_ITERATOR_HPP
