/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_DOMAIN_TENSOR_ITERATOR_HPP
#define IGUANA_DOMAIN_TENSOR_ITERATOR_HPP

#include<array>
#include<concepts>
#include<iterator>

#include "iguana/basis/tensor_product.hpp"

namespace iguana
{

/**
 * @brief Walks the elements of a tensor-structured basis.
 *
 * The iterator visits every element once, in increasing flat index, the 
 * first direction running fastest. The element data is updated in place. 
 * Every basis whose elements form a tensor grid shares this iterator,
 * as the elements depend on the knots alone.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the knot values.
 *
 * @warning The iterator holds the basis, which must outlive it.
 */
template<int d, std::floating_point T>
class TensorDomainIterator
{
public:
    /// @brief The number of parametric directions, \f$ d \f$.
    static constexpr int dimension = d;

public:
    /// @brief Starts at the first element of a given basis.
    explicit TensorDomainIterator(
        const TensorProductBSpline<d, T>& basis) noexcept;

    /// @brief The element handle, the iterator itself.
    constexpr const TensorDomainIterator& operator*() const noexcept
    { return *this; }

    /// @brief Advances to the next element.
    TensorDomainIterator& operator++() noexcept;

    /// @brief Whether the iterator has passed the last element.
    bool operator==(std::default_sentinel_t) const noexcept;

    /// @brief The element index, in [0, num_elements()).
    constexpr int index() const noexcept
    { return element_.index; }

    /// @brief The first non-zero function of each direction.
    constexpr const std::array<int, d>& first_active() const noexcept
    { return element_.first_active; }

    /// @brief The parameters at which the element starts.
    constexpr const std::array<T, d>& start() const noexcept
    { return element_.start; }

    /// @brief The parameters at which the element ends.
    constexpr const std::array<T, d>& end() const noexcept
    { return element_.end; }

private:
    /// @brief The data of the element reached, handed by the accessors.
    struct Element
    {
        /// @brief The element index, in [0, num_elements()).
        int index;

        /// @brief The first non-zero function of each direction.
        std::array<int, d> first_active;

        /// @brief The parameters at which the element starts.
        std::array<T, d> start;

        /// @brief The parameters at which the element ends.
        std::array<T, d> end;
    };

    /// @brief The element reached, refilled in place by update().
    Element element_;

    /// @brief Refills the element from the element of each direction.
    void update() noexcept;

    /// @brief The basis being walked, which outlives the iterator.
    const TensorProductBSpline<d, T>* basis_;

    /// @brief The element of every direction.
    std::array<int, d> each_;

    /// @brief The number of elements of every direction.
    std::array<int, d> counts_;
};

extern template class TensorDomainIterator<1, double>;
extern template class TensorDomainIterator<2, double>;
extern template class TensorDomainIterator<3, double>;

} // namespace iguana

#endif // IGUANA_DOMAIN_TENSOR_ITERATOR_HPP
