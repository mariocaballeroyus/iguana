/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
#define IGUANA_DOMAIN_TENSOR_DOMAIN_HPP

#include<concepts>
#include<iterator>
#include<utility>

#include "iguana/domain/tensor_iterator.hpp"

namespace iguana
{

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

extern template class TensorDomain<1, double>;
extern template class TensorDomain<2, double>;
extern template class TensorDomain<3, double>;

} // namespace iguana

#endif // IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
