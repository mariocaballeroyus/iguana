/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_CELL_CLASSIFICATION_HPP
#define IGUANA_EMBEDDING_CELL_CLASSIFICATION_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace iguana
{

/**
 * @brief Type of a cell, an element of the background grid, with respect
 *        to the physical domain
 *
 * The type is geometric only. How each type of cell is integrated is up to
 * the method built on the classification
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
 * @brief Classification of the elements of a background grid against a
 *        physical domain embedded in it
 *
 * It tells how the domain lies on the elements, with the cell type of each
 * one, and holds no geometry of the domain
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class CellClassification
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the classification from the cell type of each
     *        element
     *
     * @param cell_types Cell type of each element, in the numbering of the
     *        grid
     */
    explicit CellClassification(std::vector<CellType> cell_types);

    /// @brief Number of elements, one per cell type
    constexpr int num_elements() const noexcept
    { return static_cast<int>(cell_types_.size()); }

    /**
     * @brief Cell type of an element
     *
     * @param element Element index, in the numbering of the grid
     *
     * @pre @p element lies in [0, num_elements())
     */
    constexpr CellType cell_type(int element) const noexcept
    { return cell_types_[static_cast<std::size_t>(element)]; }

private:
    /// @brief Cell type of each element, in the numbering of the grid
    std::vector<CellType> cell_types_;
};

} // namespace iguana

#endif // IGUANA_EMBEDDING_CELL_CLASSIFICATION_HPP
