/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
#define IGUANA_DOMAIN_TENSOR_DOMAIN_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/domain/tensor_domain_iterator.hpp"
#include "iguana/patch/patch.hpp"

namespace iguana
{

/**
 * @brief Type of a cell, an element of the background patch, with respect
 *        to the physical domain
 *
 * The type is geometric only. How each type of cell is integrated is up to
 * the method built on the domain
 */
enum class CellType : std::uint8_t
{
    /// @brief Entirely outside the physical domain
    outside,

    /// @brief Entirely inside the physical domain
    inside,

    /// @brief Crossed by the boundary of the physical domain
    cut
};

/**
 * @brief Domain over the elements of a tensor-product patch
 *
 * The domain is the physical domain discretized over the elements of a
 * background patch, each of which lies outside it, inside it or across
 * its boundary. The cell types are given rather than computed, and the
 * domain holds no geometry of the solid: classifying the elements, and methods
 * that need the solid, take it alongside the domain
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
     * @brief Constructs the domain covering the whole patch
     *
     * @param patch Background patch, whose basis gives the elements
     */
    explicit TensorDomain(Patch<T, d> patch);

    /**
     * @brief Constructs the domain with the cell type of each element
     *
     * @param patch Background patch, whose basis gives the elements
     * @param cell_types Cell type of each element, in the numbering of the
     *        basis
     *
     * @throws std::invalid_argument If there is not one cell type per
     *         element
     */
    TensorDomain(Patch<T, d> patch, std::vector<CellType> cell_types);

    /// @brief Background patch
    constexpr const Patch<T, d>& patch() const noexcept
    { return patch_; }

    /// @brief Basis of the background patch, whose elements the domain has
    constexpr const TensorBSpline<T, d>& basis() const noexcept
    { return patch_.basis(); }

    /// @brief Number of elements
    constexpr int num_elements() const noexcept
    { return patch_.basis().num_elements(); }

    /**
     * @brief Cell type of an element
     *
     * @param element Element index, in the numbering of the basis
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr CellType cell_type(int element) const noexcept
    { return cell_types_[static_cast<std::size_t>(element)]; }

    /// @brief Iterator at the first element
    TensorDomainIterator<T, d> begin() const noexcept;

    /// @brief Sentinel past the last element
    constexpr std::default_sentinel_t end() const noexcept
    { return std::default_sentinel; }

private:
    /// @brief Background patch
    Patch<T, d> patch_;

    /// @brief Cell type of each element, in the numbering of the basis
    std::vector<CellType> cell_types_;
};

} // namespace iguana

#endif // IGUANA_DOMAIN_TENSOR_DOMAIN_HPP
