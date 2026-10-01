/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_bspline.hpp"

#include <span>
#include <utility>
#include <vector>

#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/// @brief Univariate bases on the knot vectors of a grid
template<std::floating_point T, std::size_t d, std::size_t... direction>
std::array<BSpline<T>, d> axes_on(const TensorGrid<T, d>& grid,
                                  std::index_sequence<direction...>)
{
    return {BSpline<T>(grid.knots(direction))...};
}

/// @brief Grid on the knot vectors of univariate bases
template<std::floating_point T, std::size_t d, std::size_t... direction>
TensorGrid<T, d> grid_of(const std::array<BSpline<T>, d>& axes,
                             std::index_sequence<direction...>)
{
    return TensorGrid<T, d>(
        std::array<KnotVector<T>, d>{axes[direction].knots()...});
}

} // namespace

template<std::floating_point T, std::size_t d>
TensorBSpline<T, d>::TensorBSpline(TensorGrid<T, d> grid)
    : grid_(std::move(grid)),
      axes_(axes_on(grid_, std::make_index_sequence<d>{})),
      num_functions_(1),
      num_active_(1)
{
    for (const BSpline<T>& axis : axes_) {
        num_functions_ *= axis.num_functions();
        num_active_ *= axis.num_active();
    }
}

template<std::floating_point T, std::size_t d>
TensorBSpline<T, d>::TensorBSpline(std::array<BSpline<T>, d> axes)
    : TensorBSpline(grid_of(axes, std::make_index_sequence<d>{}))
{
}

template<std::floating_point T, std::size_t d>
void TensorBSpline<T, d>::active_on_element(
    int element, Eigen::VectorXi& actives) const
{
    // Decode the flat element index, with the first direction fastest
    std::array<int, d> first_active{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const BSpline<T>& axis = axes_[direction];
        const int axis_element = element % axis.knots().num_elements();

        first_active[direction] = axis.first_active(axis_element);
        element /= axis.knots().num_elements();
    }

    active_on_element(first_active, actives);
}

template<std::floating_point T, std::size_t d>
void TensorBSpline<T, d>::active_on_element(
    const std::array<int, d>& first_active, Eigen::VectorXi& actives) const
{
    std::array<int, d> active_counts{};
    std::array<int, d> function_counts{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        active_counts[direction] = axes_[direction].num_active();
        function_counts[direction] = axes_[direction].num_functions();
    }

    actives.resize(num_active_);

    // Enumerate local active offsets, then flatten their global indices
    std::array<int, d> offset{};
    std::array<int, d> index{};
    int position = 0;

    do {
        for (std::size_t direction = 0; direction < d; ++direction)
            index[direction] = first_active[direction] + offset[direction];

        actives[position++] = flatten(index, function_counts);
    } while (next_lexicographic(offset, active_counts));
}

namespace
{

/**
 * @brief Forms the column-wise Kronecker product of the directional values
 *
 * Each factor has one row per active univariate function and one column per
 * point. Combining the last direction first leaves the first direction
 * running fastest in the output rows.
 *
 * @tparam d Number of directions
 * @tparam T Floating-point type of the values
 * @tparam Dst Destination matrix or writable row selection
 *
 * @param factors One value matrix per direction, with matching columns
 * @param accumulated Intermediate product for three or more directions
 * @param scratch Alternate intermediate for four or more directions
 * @param destination Pre-sized output with one row per row combination
 */
template<std::size_t d, std::floating_point T, typename Dst>
void khatri_rao_into(const std::array<const Eigen::MatrixX<T>*, d>& factors,
                     Eigen::MatrixX<T>& accumulated,
                     Eigen::MatrixX<T>& scratch, Dst&& destination)
{
    const auto combine = [](const Eigen::MatrixX<T>& factor,
                            const Eigen::MatrixX<T>& current,
                            auto&& output)
    {
        const Eigen::Index block = factor.rows();

        // Broadcast each current row over a block of factor rows
        for (Eigen::Index row = 0; row < current.rows(); ++row)
            output.middleRows(row * block, block).array()
                = factor.array().rowwise() * current.row(row).array();
    };

    if constexpr (d == 1) {
        // A single direction is already the product
        destination = *factors[0];
    }
    else if constexpr (d == 2) {
        // Two factors can write directly to the destination
        combine(*factors[0], *factors[1], destination);
    }
    else {
        // Read the last factor directly rather than copying it first
        const Eigen::MatrixX<T>& last = *factors[d - 1];
        const Eigen::MatrixX<T>& previous = *factors[d - 2];
        accumulated.resize(last.rows() * previous.rows(), last.cols());
        combine(previous, last, accumulated);

        // Only interior directions need an intermediate buffer
        for (std::size_t dir = d - 3; dir > 0; --dir) {
            const Eigen::MatrixX<T>& factor = *factors[dir];

            scratch.resize(accumulated.rows() * factor.rows(),
                           factor.cols());
            combine(factor, accumulated, scratch);
            accumulated.swap(scratch);
        }

        // The first direction writes the final result directly
        combine(*factors[0], accumulated, destination);
    }
}

} // namespace

template<std::floating_point T, std::size_t d>
void TensorBSpline<T, d>::eval_on_element(
    const std::array<int, d>& first_active,
    const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values) const
{
    const Eigen::Index num_points = points.rows();

    // Reuse the output buffer when its shape is unchanged
    values.resize(num_active_, num_points);

    // Evaluate the active univariate functions in each direction
    std::array<Eigen::MatrixX<T>, d> axis_values;

    for (std::size_t direction = 0; direction < d; ++direction) {
        // Coordinates of one direction are contiguous in a column
        const std::span<const T> coords(points.col(direction).data(),
                                        num_points);

        axes_[direction].eval_on_element(first_active[direction], coords,
                                         axis_values[direction]);
    }

    // The column-wise Kronecker product combines matching point columns
    std::array<const Eigen::MatrixX<T>*, d> factors{};

    for (std::size_t direction = 0; direction < d; ++direction)
        factors[direction] = &axis_values[direction];

    Eigen::MatrixX<T> accumulated;
    Eigen::MatrixX<T> scratch;

    khatri_rao_into(factors, accumulated, scratch, values);
}

template<std::floating_point T, std::size_t d>
void TensorBSpline<T, d>::grad_on_element(
    const std::array<int, d>& first_active,
    const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values,
    std::array<Eigen::MatrixX<T>, d>& gradients) const
{
    const Eigen::Index num_points = points.rows();

    values.resize(num_active_, num_points);

    // Values and first derivatives of the active univariate functions
    std::array<std::vector<Eigen::MatrixX<T>>, d> axis_derivs;

    for (std::size_t direction = 0; direction < d; ++direction) {
        const std::span<const T> coords(points.col(direction).data(),
                                        num_points);

        axes_[direction].derivs_on_element(first_active[direction], coords,
                                           1, axis_derivs[direction]);
    }

    std::array<const Eigen::MatrixX<T>*, d> factors{};

    for (std::size_t direction = 0; direction < d; ++direction)
        factors[direction] = &axis_derivs[direction][0];

    Eigen::MatrixX<T> accumulated;
    Eigen::MatrixX<T> scratch;

    khatri_rao_into(factors, accumulated, scratch, values);

    // Each gradient swaps the factor of its direction for its derivative
    for (std::size_t direction = 0; direction < d; ++direction) {
        factors[direction] = &axis_derivs[direction][1];
        gradients[direction].resize(num_active_, num_points);
        khatri_rao_into(factors, accumulated, scratch, gradients[direction]);
        factors[direction] = &axis_derivs[direction][0];
    }
}

template<std::floating_point T, std::size_t d>
void TensorBSpline<T, d>::hess_on_element(
    const std::array<int, d>& first_active,
    const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values,
    std::array<Eigen::MatrixX<T>, d>& gradients,
    std::array<Eigen::MatrixX<T>, d * (d + 1) / 2>& hessians) const
{
    const Eigen::Index num_points = points.rows();

    // Values and first two derivatives of the active univariate functions
    std::array<std::vector<Eigen::MatrixX<T>>, d> axis_derivs;

    for (std::size_t direction = 0; direction < d; ++direction) {
        const std::span<const T> coords(points.col(direction).data(),
                                        num_points);

        axes_[direction].derivs_on_element(first_active[direction], coords,
                                           2, axis_derivs[direction]);
    }

    // How often the factor of each direction is differentiated
    std::array<int, d> orders{};

    Eigen::MatrixX<T> accumulated;
    Eigen::MatrixX<T> scratch;

    const auto product = [&](Eigen::MatrixX<T>& result) {
        std::array<const Eigen::MatrixX<T>*, d> factors{};

        for (std::size_t direction = 0; direction < d; ++direction)
            factors[direction] = &axis_derivs[direction][orders[direction]];

        result.resize(num_active_, num_points);
        khatri_rao_into(factors, accumulated, scratch, result);
    };

    product(values);

    for (std::size_t direction = 0; direction < d; ++direction) {
        orders[direction] = 1;
        product(gradients[direction]);
        orders[direction] = 2;
        product(hessians[direction]);
        orders[direction] = 0;
    }

    // Mixed pairs follow the pure ones, in lexicographic order
    std::size_t pair = d;

    for (std::size_t first = 0; first < d; ++first) {
        for (std::size_t second = first + 1; second < d; ++second) {
            orders[first] = orders[second] = 1;
            product(hessians[pair++]);
            orders[first] = orders[second] = 0;
        }
    }
}

template class TensorBSpline<double, 1>;
template class TensorBSpline<double, 2>;
template class TensorBSpline<double, 3>;

} // namespace iguana
