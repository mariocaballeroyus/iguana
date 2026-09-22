/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "patch.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace iguana
{

using Eigen::placeholders::all;

template<std::floating_point T, std::size_t d>
Patch<T, d>::Patch(TensorBSpline<T, d> basis,
                   PointMatrix<T> coefficients)
    : basis_(std::move(basis)),
      coefficients_(std::move(coefficients))
{
    if (coefficients_.rows() != basis_.num_functions())
        throw std::invalid_argument("Patch: "
                                    "there must be one control point per "
                                    "basis function");
}

template<std::floating_point T, std::size_t d>
void Patch<T, d>::position_on_element(const Eigen::VectorXi& actives,
                                      const Eigen::MatrixX<T>& values,
                                      PointMatrix<T>& positions) const
{
    // Reuse the output buffer when its shape is unchanged
    positions.resize(values.cols(), 3);

    // Gather the control points of the active functions, then weight each
    // of them by the value of its function at every point
    positions.noalias() = values.transpose() * coefficients_(actives, all);
}

template<std::floating_point T, std::size_t d>
Patch<T, 1> Patch<T, d>::isocurve(std::size_t direction,
                                  const std::array<int, d - 1>& lines) const
{
    // Strides of the flattened numbering, first direction fastest
    std::array<int, d> stride{};
    int step = 1;

    for (std::size_t dir = 0; dir < d; ++dir) {
        stride[dir] = step;
        step *= basis_.axis(dir).num_functions();
    }

    // Weights of the pinned directions, empty when nothing is pinned
    std::array<Eigen::MatrixX<T>, d - 1> weights;
    std::array<int, d - 1> firsts{};
    std::array<int, d - 1> active_counts{};
    std::array<int, d - 1> strides{};
    std::size_t pin = 0;

    for (std::size_t dir = 0; dir < d; ++dir) {
        if (dir == direction)
            continue;

        const BSpline<T>& axis = basis_.axis(dir);
        const int line = lines[pin];

        // The last line closes an element, every other one starts it
        const bool last = line == axis.num_elements();
        const int element = last ? line - 1 : line;
        const T parameter = last ? axis.element_end(element)
                                 : axis.element_start(element);

        firsts[pin] = axis.first_active(element);
        active_counts[pin] = axis.num_active();
        strides[pin] = stride[dir];

        axis.eval_on_element(firsts[pin], std::span<const T>(&parameter, 1),
                             weights[pin]);
        ++pin;
    }

    // Contract the net, leaving one control point per function
    const BSpline<T>& axis = basis_.axis(direction);
    const int num_functions = axis.num_functions();
    const int curve_stride = stride[direction];

    PointMatrix<T> curve = PointMatrix<T>::Zero(num_functions, 3);
    std::array<int, d - 1> offset{};
    bool remaining = true;

    while (remaining) {
        int first = 0;
        T weight = T(1);

        for (std::size_t k = 0; k < d - 1; ++k) {
            first += (firsts[k] + offset[k]) * strides[k];
            weight *= weights[k](offset[k], 0);
        }

        for (int function = 0; function < num_functions; ++function)
            curve.row(function) +=
                weight * coefficients_.row(first + function * curve_stride);

        // Advance the active functions in lexicographic order
        remaining = false;

        for (std::size_t k = 0; k < d - 1; ++k) {
            if (++offset[k] < active_counts[k]) {
                remaining = true;
                break;
            }

            offset[k] = 0;
        }
    }

    return Patch<T, 1>(TensorBSpline<T, 1>(std::array<BSpline<T>, 1>{axis}),
                       std::move(curve));
}

template<std::floating_point T, std::size_t d>
std::array<std::vector<Patch<T, 1>>, d> Patch<T, d>::isocurves() const
{
    std::array<std::vector<Patch<T, 1>>, d> groups;

    for (std::size_t direction = 0; direction < d; ++direction) {
        // n elements are bounded by n + 1 knot lines
        std::array<int, d - 1> counts{};
        std::size_t pin = 0;
        int total = 1;

        for (std::size_t dir = 0; dir < d; ++dir) {
            if (dir == direction)
                continue;

            counts[pin] = basis_.axis(dir).num_elements() + 1;
            total *= counts[pin];
            ++pin;
        }

        groups[direction].reserve(static_cast<std::size_t>(total));

        // Walk the knot lines, the first pinned direction fastest
        std::array<int, d - 1> lines{};
        bool remaining = true;

        while (remaining) {
            groups[direction].push_back(isocurve(direction, lines));
            remaining = false;

            for (std::size_t k = 0; k < d - 1; ++k) {
                if (++lines[k] < counts[k]) {
                    remaining = true;
                    break;
                }

                lines[k] = 0;
            }
        }
    }

    return groups;
}

template class Patch<double, 1>;
template class Patch<double, 2>;
template class Patch<double, 3>;

} // namespace iguana
