/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_bspline.hpp"

#include <span>
#include <utility>

#include "iguana/multi_index.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorBSpline<T, d>::TensorBSpline(std::array<BSpline<T>, d> axes)
    : axes_(std::move(axes)),
      num_functions_(1),
      num_elements_(1),
      num_active_(1)
{
    for (const BSpline<T>& axis : axes_) {
        num_functions_ *= axis.num_functions();
        num_elements_ *= axis.num_elements();
        num_active_ *= axis.num_active();
    }
}

template<std::floating_point T, std::size_t d>
void TensorBSpline<T, d>::active_on_element(
    int element, Eigen::VectorXi& actives) const
{
    // Decode the flat element index, with the first direction fastest
    std::array<int, d> first_active{};
    std::array<int, d> active_counts{};
    std::array<int, d> function_counts{};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const BSpline<T>& axis = axes_[direction];
        const int axis_element = element % axis.num_elements();

        first_active[direction] = axis.first_active(axis_element);
        active_counts[direction] = axis.num_active();
        function_counts[direction] = axis.num_functions();
        element /= axis.num_elements();
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

template class TensorBSpline<double, 1>;
template class TensorBSpline<double, 2>;
template class TensorBSpline<double, 3>;

} // namespace iguana
