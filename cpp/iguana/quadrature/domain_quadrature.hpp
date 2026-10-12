/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_DOMAIN_QUADRATURE_HPP
#define IGUANA_QUADRATURE_DOMAIN_QUADRATURE_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/embedding/cell_classification.hpp"
#include "iguana/quadrature/quadrature.hpp"

namespace iguana
{

/**
 * @brief Quadrature over the elements of a grid
 *
 * Its groups are the integrated elements, which may hold different numbers
 * of points, so that cut elements can carry rules of their own, and
 * elements that are not integrated are left out
 *
 * A quadrature is filled one cell type at a time, each with a rule of its
 * own, such as Gauss-Legendre on inside cells and moment fitting on cut
 * cells
 *
 * The weights include the measure of their element in parameter space but
 * not the Jacobian of the geometry map, which the assembly applies
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class DomainQuadrature : public Quadrature<T, d>
{
public:
    /// @brief Constructs an empty quadrature, which integrates nothing
    DomainQuadrature() = default;

    /**
     * @brief Constructs the quadrature from its flat arrays
     *
     * @param elements Index of each integrated element, in the numbering of
     *        the grid the quadrature is built on
     * @param offsets First point of each integrated element, followed by
     *        the number of points, with size num_elements + 1
     * @param points Points in parameter space, with size (num_points, d)
     * @param weights Weights, one per point
     *
     * @throws std::invalid_argument If the points do not have d columns, if
     *         there is not one weight per point, if there is not one offset
     *         more than elements, or if the offsets do not run from zero to
     *         the number of points without decreasing
     */
    DomainQuadrature(Eigen::VectorXi elements, Eigen::VectorXi offsets,
                     Eigen::MatrixX<T> points, Eigen::VectorX<T> weights);

    /**
     * @brief Fills the cells of one type with a rule
     *
     * The rule fills the quadrature of every cell of the type on the
     * reference cell, which BoxRule::map_to_parameter_space() places on the
     * cell and which is appended in increasing element index after the
     * elements already held. If it throws, the quadrature is left unchanged
     *
     * @tparam Grid Grid with num_elements() whose walk gives each
     *         element its index(), start() and end(), as TensorGrid and
     *         HierarchicalGrid do
     * @tparam Rule Rule with fill_to_reference_space(start, end, points,
     *         weights), as BoxRule states it. The number of points may
     *         differ from one cell to another
     *
     * @param grid Grid whose elements are the cells
     * @param classification Cell type of each element of the grid
     * @param cell_type Type of the cells to fill
     * @param rule Rule giving the points and weights of each cell
     *
     * @throws std::invalid_argument If the classification does not have one
     *         cell type per element of the grid, if an element already held
     *         lies outside the grid, or if a cell of this type is already
     *         held
     */
    template<typename Grid, typename Rule>
    void fill(const Grid& grid,
              const CellClassification<T, d>& classification,
              CellType cell_type, const Rule& rule);

    /// @brief Number of integrated elements
    constexpr int num_elements() const noexcept
    { return static_cast<int>(elements_.size()); }

    /// @brief Index of each integrated element, in the grid numbering
    constexpr const Eigen::VectorXi& elements() const noexcept
    { return elements_; }

private:
    using Quadrature<T, d>::offsets_;
    using Quadrature<T, d>::points_;
    using Quadrature<T, d>::weights_;

    /// @brief Index of each integrated element, in the grid numbering
    Eigen::VectorXi elements_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_DOMAIN_QUADRATURE_HPP
