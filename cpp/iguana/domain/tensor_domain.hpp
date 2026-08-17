/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
#define IGUANA_DOMAIN_TENSOR_DOMAIN_HPP

#include<array>
#include<concepts>
#include<iterator>
#include<utility>

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

/**
 * @brief The domain of a tensor-structured basis.
 *
 * A domain owns the basis whose elements are visited, and hands out iterators 
 * to loop through those elements. It outlives every iterator it hands out.
 * Every basis whose elements form a tensor grid shares this domain, as the
 * elements depend on the knots alone.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the knot values.
 */
template<int d, std::floating_point T>
class TensorDomain
{
public:
    /// @brief The number of parametric directions, \f$ d \f$.
    static constexpr int dimension = d;

public:
    /// @brief Constructs the domain of a given basis.
    explicit TensorDomain(TensorProductBSpline<d, T> basis)
        : basis_(std::move(basis))
    {
    }

    /// @brief The basis whose elements the domain holds.
    constexpr const TensorProductBSpline<d, T>& basis() const noexcept
    { return basis_; }

    /// @brief The number of elements of the domain.
    constexpr int num_elements() const noexcept
    { return basis_.num_elements(); }

    /// @brief An iterator at the first element, borrowing the basis.
    TensorDomainIterator<d, T> begin() const noexcept;

    /// @brief The sentinel past the last element.
    constexpr std::default_sentinel_t end() const noexcept
    { return std::default_sentinel; }

private:
    /// @brief The basis whose elements the domain holds.
    TensorProductBSpline<d, T> basis_;
};

extern template class TensorDomainIterator<1, double>;
extern template class TensorDomainIterator<2, double>;
extern template class TensorDomainIterator<3, double>;

extern template class TensorDomain<1, double>;
extern template class TensorDomain<2, double>;
extern template class TensorDomain<3, double>;

} // namespace iguana

#endif // IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
