/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_FSPACE_FUNCTION_SPACE_HPP
#define IGUANA_FSPACE_FUNCTION_SPACE_HPP

#include <concepts>
#include <cstddef>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/embedding/embedding.hpp"
#include "iguana/fspace/dof_map.hpp"

namespace iguana
{

/**
 * @brief Standard space of a basis on a physical domain, with one degree of
 *        freedom per function active on it
 *
 * A function active on a cell that is not outside the physical domain gets
 * a degree of freedom, numbered in increasing function index, and one
 * supported on outside cells alone gets none. On a cell that is not
 * outside, the k-th degree of freedom belongs to the k-th function of
 * TensorBSpline::active_on_element(), so that the basis values pair with
 * them directly. Outside cells list no degrees of freedom
 *
 * The space covers the cells its embedding does not mark outside, so a
 * method integrating a smaller region, such as the inside cells alone,
 * passes the embedding of that region
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class FunctionSpace
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the space of a basis on the physical domain that an
     *        embedding places on its elements
     *
     * @param basis Basis whose functions span the space
     * @param embedding Cell type of each element of the basis domain. With
     *        every cell inside, each function gets the degree of freedom of
     *        its own index
     *
     * @throws std::invalid_argument If the embedding does not have one cell
     *         type per element
     */
    FunctionSpace(TensorBSpline<T, d> basis,
                  const Embedding<T, d>& embedding);

    /// @brief Basis whose functions span the space
    constexpr const TensorBSpline<T, d>& basis() const noexcept
    { return basis_; }

    /// @brief Degrees of freedom of each element of the basis domain
    constexpr const DofMap& dof_map() const noexcept
    { return dof_map_; }

private:
    /// @brief Basis whose functions span the space
    TensorBSpline<T, d> basis_;

    /// @brief Degrees of freedom of each element, built from the basis
    DofMap dof_map_;
};

} // namespace iguana

#endif // IGUANA_FSPACE_FUNCTION_SPACE_HPP
