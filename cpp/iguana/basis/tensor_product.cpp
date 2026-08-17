/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "tensor_product.hpp"

#include<span>
#include<utility>

#include "iguana/multi_index.hpp"

namespace iguana
{

using Eigen::placeholders::all;
using Eigen::seqN;

template<int d, std::floating_point T>
TensorProductBSpline<d, T>::TensorProductBSpline(
    std::array<BSpline<T>, d> axes)
    : axes_(std::move(axes))
{
    num_functions_ = 1;
    num_elements_ = 1;
    num_active_ = 1;

    for (const BSpline<T>& basis : axes_) {
        num_functions_ *= basis.num_functions();
        num_elements_ *= basis.num_elements();
        num_active_ *= basis.num_active();
    }
}

// Elements --------------------------------------------------------------------

namespace
{

/**
 * @brief The element of every direction of a given element.
 *
 * @tparam d Number of directions.
 * @tparam T Floating-point type of the knot values.
 *
 * @param axes The univariate bases, one per parametric direction.
 * @param element The element index, below the product of the counts.
 *
 * @return The element of every direction.
 */
template<std::size_t d, std::floating_point T>
std::array<int, d> element_of(const std::array<BSpline<T>, d>& axes,
                              int element) noexcept
{
    std::array<int, d> counts{};

    for (std::size_t k = 0; k < d; ++k)
        counts[k] = axes[k].num_elements();

    return unflatten(element, counts);
}

} // namespace

template<int d, std::floating_point T>
std::array<int, d> TensorProductBSpline<d, T>::first_active(
    int element) const noexcept
{
    const std::array<int, d> each = element_of(axes_, element);

    std::array<int, d> first{};

    for (int k = 0; k < d; ++k)
        first[k] = axes_[k].first_active(each[k]);

    return first;
}

template<int d, std::floating_point T>
std::array<T, d> TensorProductBSpline<d, T>::element_start(
    int element) const noexcept
{
    const std::array<int, d> each = element_of(axes_, element);

    std::array<T, d> start{};

    for (int k = 0; k < d; ++k)
        start[k] = axes_[k].element_start(each[k]);

    return start;
}

template<int d, std::floating_point T>
std::array<T, d> TensorProductBSpline<d, T>::element_end(
    int element) const noexcept
{
    const std::array<int, d> each = element_of(axes_, element);

    std::array<T, d> end{};

    for (int k = 0; k < d; ++k)
        end[k] = axes_[k].element_end(each[k]);

    return end;
}

template<int d, std::floating_point T>
void TensorProductBSpline<d, T>::active_on_element(
    int element, Eigen::VectorXi& actives) const
{
    // Actives buffer, reused across elements
    actives.resize(num_active_);

    // Lowest non-zero function of each direction
    const std::array<int, d> first = first_active(element);

    // Bounds of the multi-index, one non-zero function per direction,
    // and the counts the whole basis is numbered by
    std::array<int, d> bounds{};
    std::array<int, d> counts{};

    for (int k = 0; k < d; ++k) {
        bounds[k] = axes_[k].num_active();
        counts[k] = axes_[k].num_functions();
    }

    // Offsets from the lowest function, advanced by next_lexicographic
    std::array<int, d> offset{};
    std::array<int, d> index{};
    int position = 0;

    do {
        for (int k = 0; k < d; ++k)
            index[k] = first[k] + offset[k];

        // The rows of the evaluation follow the same order
        actives[position++] = flatten(index, counts);
    } while (next_lexicographic(offset, bounds));
}

// Evaluation ------------------------------------------------------------------

namespace
{

/**
 * @brief Combines one matrix per direction into a Khatri-Rao product.
 *
 * @tparam d Number of directions.
 * @tparam T Floating-point type of the values.
 * @tparam Dst Type of the destination, which is either a whole matrix or
 *         a selection of the rows of one.
 *
 * @param factors The matrix of every direction, all of them holding the
 *        same number of columns.
 * @param accumulated Scratch, holding the product on return.
 * @param scratch Further scratch, reused between the directions.
 * @param destination Filled with the product, one row per combination of the
 *        rows of the factors.
 */
template<std::size_t d, std::floating_point T, typename Dst>
void khatri_rao_into(const std::array<const Eigen::MatrixX<T>*, d>& factors,
                     Eigen::MatrixX<T>& accumulated,
                     Eigen::MatrixX<T>& scratch, Dst&& destination)
{
    if constexpr (d == 1) {
        // A single direction is already the product
        destination = *factors[0];
        return;
    }
    else {
        // The last one first so that the first one ends up running fastest
        accumulated = *factors[d - 1];

        for (int k = d - 2; k >= 0; --k) {
            const Eigen::MatrixX<T>& factor = *factors[k];

            const Eigen::Index rows = accumulated.rows();
            const Eigen::Index block = factor.rows();

            if (k > 0) {
                scratch.resize(rows * block, factor.cols());

                // Each accumulated row broadcasts over a whole block
                for (Eigen::Index r = 0; r < rows; ++r)
                    scratch.middleRows(r * block, block).array()
                        = factor.array().rowwise()
                          * accumulated.row(r).array();

                accumulated.swap(scratch);
            }
            else {
                // The last direction writes the destination itself
                for (Eigen::Index r = 0; r < rows; ++r)
                    destination.middleRows(r * block, block).array()
                        = factor.array().rowwise()
                          * accumulated.row(r).array();
            }
        }
    }
}

} // namespace

template<int d, std::floating_point T>
void TensorProductBSpline<d, T>::eval_on_element(
    const std::array<int, d>& first_active, const Eigen::MatrixX<T>& points,
    Eigen::MatrixX<T>& values) const
{
    const Eigen::Index num_pts = points.rows();

    // Values buffer, reused across elements
    values.resize(num_active_, num_pts);

    // Univariate values of every direction
    std::array<Eigen::MatrixX<T>, d> axis_values;

    for (int k = 0; k < d; ++k) {
        // Coordinates of the direction, contiguous as they are a column
        const std::span<const T> coords(points.col(k).data(), num_pts);

        axes_[k].eval_on_element(first_active[k], coords, axis_values[k]);
    }

    // Khatri-Rao product, column-wise Kronecker of the directions
    std::array<const Eigen::MatrixX<T>*, d> factors{};

    for (int k = 0; k < d; ++k)
        factors[k] = &axis_values[k];

    Eigen::MatrixX<T> accumulated;
    Eigen::MatrixX<T> scratch;

    khatri_rao_into(factors, accumulated, scratch, values);
}

template<int d, std::floating_point T>
void TensorProductBSpline<d, T>::eval_ders_on_element(
    const std::array<int, d>& first_active, const Eigen::MatrixX<T>& points,
    int order, std::vector<Eigen::MatrixX<T>>& values) const
{
    const Eigen::Index num_pts = points.rows();

    // Univariate values and derivatives of every direction
    std::array<std::vector<Eigen::MatrixX<T>>, d> axis_ders;

    for (int k = 0; k < d; ++k) {
        // Coordinates of the direction, contiguous as they are a column
        const std::span<const T> coords(points.col(k).data(), num_pts);

        axes_[k].eval_ders_on_element(first_active[k], coords, order,
                                      axis_ders[k]);
    }

    // One buffer per order, the slots of a function being consecutive
    values.resize(static_cast<std::size_t>(order) + 1);

    for (int k = 0; k <= order; ++k)
        values[static_cast<std::size_t>(k)].resize(
            num_active_ * num_slots(k), num_pts);

    // Khatri-Rao product, column-wise Kronecker of the directions
    std::array<const Eigen::MatrixX<T>*, d> factors{};

    Eigen::MatrixX<T> accumulated;
    Eigen::MatrixX<T> scratch;

    // A slot is written into the rows it occupies in every function
    // Order 0, every direction taking its own values
    for (int k = 0; k < d; ++k)
        factors[k] = &axis_ders[k][0];

    khatri_rao_into(factors, accumulated, scratch, values[0]);

    if (order >= 1) {
        const int slots = num_slots(1);

        // Order 1, one direction at a time taking its first derivative
        for (int k = 0; k < d; ++k) {
            factors[k] = &axis_ders[k][1];
            const auto rows = seqN(k, num_active_, slots);

            khatri_rao_into(factors, accumulated, scratch,
                            values[1](rows, all));

            factors[k] = &axis_ders[k][0];
        }
    }

    if (order >= 2) {
        const int slots = num_slots(2);

        // Order 2, the pure derivatives in direction order
        for (int k = 0; k < d; ++k) {
            factors[k] = &axis_ders[k][2];
            const auto rows = seqN(k, num_active_, slots);

            khatri_rao_into(factors, accumulated, scratch,
                            values[2](rows, all));

            factors[k] = &axis_ders[k][0];
        }

        // Then the mixed ones, in the lexicographic order of the pairs
        int slot = d;

        for (int k = 0; k < d; ++k) {
            for (int l = k + 1; l < d; ++l) {
                factors[k] = &axis_ders[k][1];
                factors[l] = &axis_ders[l][1];
                const auto rows = seqN(slot++, num_active_, slots);

                khatri_rao_into(factors, accumulated, scratch,
                                values[2](rows, all));

                factors[k] = &axis_ders[k][0];
                factors[l] = &axis_ders[l][0];
            }
        }
    }
}

template class TensorProductBSpline<1, double>;
template class TensorProductBSpline<2, double>;
template class TensorProductBSpline<3, double>;

} // namespace iguana
