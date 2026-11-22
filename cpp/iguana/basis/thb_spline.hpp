/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_BASIS_THB_SPLINE_HPP
#define IGUANA_BASIS_THB_SPLINE_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/grid/hierarchical_grid.hpp"

namespace iguana
{

/**
 * @brief Truncated hierarchical B-spline basis on a hierarchical grid
 *
 * Each level of the grid carries a tensor-product B-spline basis of the
 * same degrees on its elements, the tensor basis of a level refining that
 * of the level before. A function of a level belongs to the basis when its
 * support lies in the domain of its level but not in that of the next, the
 * selection of Kraft, and the functions are numbered level by level and,
 * within a level, in increasing index of the level basis
 *
 * A function of a level is truncated: written in the basis of each finer
 * level in turn, it drops the components along the functions of that level
 * that belong to the basis. The truncated functions still partition unity,
 * and a spline of the coarse tensor space keeps its tensor coefficients at
 * every level, so that the coefficients of a function of the basis are
 * those of its level
 *
 * On an element, the basis evaluates the tensor basis of the element's
 * level and combines it: the functions of coarser levels non-zero on the
 * element are written in the functions of the level non-zero on it, level
 * by level through the refinement of the knot vectors, with the components
 * along the functions of the basis dropped at each level. The combination
 * is rebuilt at each evaluation
 *
 * Several bases of different degrees on one grid, such as the velocities
 * and the pressure of a divergence-conforming space, select their functions
 * by the same domains, so that the divergence of a velocity of the
 * hierarchical space lies in the hierarchical pressure space
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class THBSpline
{
public:
    /// @brief Floating-point type
    using Scalar = T;

    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the basis of given degrees on a hierarchical grid
     *
     * The knot vector of each level and direction takes the knot lines of
     * the level grid, with the interior multiplicities of its knots and the
     * ends repeated once per degree plus one, open
     *
     * @param grid Hierarchical grid, whose coarse knot vectors must be open
     * @param degrees Degree of the basis along each direction
     *
     * @throws std::invalid_argument If a degree is negative or above
     *         BSpline::max_degree, or if a coarse knot vector of the grid
     *         is not open
     */
    THBSpline(HierarchicalGrid<T, d> grid, std::array<int, d> degrees);

    /// @brief Grid, whose active elements the basis is defined on
    constexpr const HierarchicalGrid<T, d>& grid() const noexcept
    { return grid_; }

    /// @brief Degree of the basis along each direction
    constexpr const std::array<int, d>& degrees() const noexcept
    { return degrees_; }

    /// @brief Number of levels, those of the grid
    constexpr int num_levels() const noexcept
    { return static_cast<int>(levels_.size()); }

    /**
     * @brief Tensor-product basis of a level
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels())
     */
    constexpr const TensorBSpline<T, d>& level(int level) const noexcept
    { return levels_[static_cast<std::size_t>(level)]; }

    /**
     * @brief Functions of a level in the basis, in increasing index of the
     *        level basis
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels())
     */
    constexpr const std::vector<int>& active_functions(int level) const
        noexcept
    { return active_[static_cast<std::size_t>(level)]; }

    /**
     * @brief Index of the first function of a level, that of the level
     *        after the last one being num_functions()
     *
     * @param level Level index
     *
     * @pre @p level lies in [0, num_levels()]
     */
    constexpr int offset(int level) const noexcept
    { return offsets_[static_cast<std::size_t>(level)]; }

    /// @brief Number of functions over all levels
    constexpr int num_functions() const noexcept
    { return offsets_.back(); }

    /**
     * @brief Functions that are non-zero on an active element, those of
     *        its level and the coarser ones whose truncation leaves
     *        something on it, level by level and in increasing index within
     *        a level
     *
     * @param element Element index, in the numbering of the grid
     * @param actives Output, resized to the number of functions
     *
     * @pre @p element lies in [0, grid().num_elements())
     */
    void active_on_element(int element, Eigen::VectorXi& actives) const;

    /**
     * @brief Evaluates the non-zero functions on an active element
     *
     * @param element Element index, in the numbering of the grid
     * @param points Evaluation points, one per row and one coordinate per
     *        column, with size (num_points, dimension)
     * @param values Output with a row per function of active_on_element()
     *        and a column per point. It is resized when necessary
     *
     * @pre @p element lies in [0, grid().num_elements()), @p points has
     *      dimension columns, and every point lies inside that element
     */
    void eval_on_element(int element, const Eigen::MatrixX<T>& points,
                         Eigen::MatrixX<T>& values) const;

    /**
     * @brief Evaluates the non-zero functions on an active element and
     *        their gradients
     *
     * @param element Element index, in the numbering of the grid
     * @param points Evaluation points, with size (num_points, dimension)
     * @param values Output filled as eval_on_element() fills it
     * @param gradients Output with one matrix per direction, of the size of
     *        @p values, holding the derivatives along it. They are resized
     *        when necessary
     *
     * @pre @p element lies in [0, grid().num_elements()), @p points has
     *      dimension columns, and every point lies inside that element
     */
    void grad_on_element(int element, const Eigen::MatrixX<T>& points,
                         Eigen::MatrixX<T>& values,
                         std::array<Eigen::MatrixX<T>, d>& gradients) const;

    /**
     * @brief Evaluates the non-zero functions on an active element, their
     *        gradients and their Hessians
     *
     * The Hessians are ordered pure first, then mixed in lexicographic
     * order of their directions: xx, yy, zz, xy, xz, yz
     *
     * @param element Element index, in the numbering of the grid
     * @param points Evaluation points, with size (num_points, dimension)
     * @param values Output filled as eval_on_element() fills it
     * @param gradients Output filled as grad_on_element() fills it
     * @param hessians Output with one matrix per pair of directions, of the
     *        size of @p values. They are resized when necessary
     *
     * @pre @p element lies in [0, grid().num_elements()), @p points has
     *      dimension columns, and every point lies inside that element
     */
    void hess_on_element(
        int element, const Eigen::MatrixX<T>& points,
        Eigen::MatrixX<T>& values, std::array<Eigen::MatrixX<T>, d>& gradients,
        std::array<Eigen::MatrixX<T>, d * (d + 1) / 2>& hessians) const;

    /**
     * @brief Coefficients in the basis of splines of the coarse tensor
     *        space, which the basis spans
     *
     * The tensor coefficients of a spline at a level are the refinement of
     * those at the level before, and the truncated functions preserve them:
     * the coefficient of a function of the basis is the tensor coefficient
     * of its level
     *
     * @param coarse Coefficients of the splines in the tensor basis of
     *        level 0, one row per function and one column per spline
     * @return Coefficients with one row per function of the basis
     *
     * @throws std::invalid_argument If there is not one row per function of
     *         the coarse basis
     */
    Eigen::MatrixX<T> coefficients(const Eigen::MatrixX<T>& coarse) const;

    /**
     * @brief Functions on the two sides of the parametric domain normal to
     *        a direction, those whose index along it is the first or the
     *        last of their level, which alone are non-zero there
     *
     * @param direction Direction normal to the sides
     * @return Function indices, increasing
     *
     * @pre @p direction lies in [0, dimension)
     */
    Eigen::VectorXi boundary_functions(std::size_t direction) const;

    /// @brief Whether two bases have the same grid and degrees
    bool operator==(const THBSpline& other) const;

private:
    /**
     * @brief Coefficients of the functions of a level in those of the
     *        next, along one direction, which refines each function into a
     *        band of consecutive ones
     */
    struct Refinement
    {
        /// @brief First fine function of the band of each coarse function
        Eigen::VectorXi first;

        /// @brief Coefficients of the band of each coarse function, a
        ///        column per coarse function and a row per fine function
        ///        of the band from its first, zero beyond the band
        Eigen::MatrixX<T> coefficients;
    };

    /**
     * @brief Functions of a level non-zero on an active element: their
     *        positions among the functions of the level basis non-zero on
     *        the element's ancestor of the level, and their indices
     */
    struct Selection
    {
        /// @brief First active function of each direction of the ancestor,
        ///        in the level basis
        std::array<int, d> first_active;

        /// @brief Positions of the functions among those of the level basis
        ///        non-zero on the ancestor, increasing
        std::vector<int> local;

        /// @brief Indices of the functions in the basis
        std::vector<int> global;
    };

    /// @brief Functions non-zero on an active element and how they combine
    ///        the functions of its level
    struct Local
    {
        /// @brief First active function of each direction of the element,
        ///        in the basis of its level
        std::array<int, d> first_active;

        /// @brief Indices of the functions in the basis
        Eigen::VectorXi global;

        /// @brief Combination giving the functions out of those of the
        ///        level basis non-zero on the element, a row per function
        ///        and a column per function of the level basis
        Eigen::MatrixX<T> extraction;
    };

    /// @brief Selection of each level up to that of an active element,
    ///        of the functions whose supports hold it
    std::vector<Selection> selections(int element) const;

    /**
     * @brief Functions non-zero on an active element: the selected ones,
     *        written in the functions of its level, less those the
     *        truncation leaves nothing of on it
     */
    Local localize(int element) const;

    /// @brief Grid, whose active elements the basis is defined on
    HierarchicalGrid<T, d> grid_;

    /// @brief Degree of the basis along each direction
    std::array<int, d> degrees_;

    /// @brief Tensor-product basis of each level
    std::vector<TensorBSpline<T, d>> levels_;

    /// @brief Refinement of each level but the last into the next, along
    ///        each direction
    std::vector<std::array<Refinement, d>> refinements_;

    /// @brief Functions of each level in the basis, in increasing index
    std::vector<std::vector<int>> active_;

    /// @brief Index of the first function of each level, followed by the
    ///        number of functions
    std::vector<int> offsets_;
};

} // namespace iguana

#endif // IGUANA_BASIS_THB_SPLINE_HPP
