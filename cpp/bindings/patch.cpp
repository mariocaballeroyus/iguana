/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using CurvePatch = Patch<double, 1>;
using VolumePatch = Patch<double, 3>;

/**
 * @brief Isocurve running along one direction, at knot lines of the others
 *
 * Only the control points are formed, by contracting the control net
 * against the values of the pinned directions at their knot lines
 *
 * @param patch Patch the curve is read from
 * @param direction Direction the curve runs along
 * @param lines Knot line pinning each remaining direction, in increasing
 *        order of direction
 *
 * @return Curve patch of the isocurve
 *
 * @pre Every line lies in [0,num_elements] of its own axis, which
 *      isocurves() ensures
 */
CurvePatch isocurve(const VolumePatch& patch,
                    std::size_t direction,
                    const std::array<int, 2>& lines)
{
    const TensorBSpline<double, 3>& basis = patch.basis();
    const PointMatrix<double>& coefficients = patch.coefficients();

    // Strides of the flattened numbering, first direction fastest
    std::array<int, 3> stride{};
    int step = 1;

    for (std::size_t dir = 0; dir < 3; ++dir) {
        stride[dir] = step;
        step *= basis.axis(dir).num_functions();
    }

    // Weights of the pinned directions at their knot line
    std::array<Eigen::MatrixXd, 2> weights;
    std::array<int, 2> firsts{};
    std::array<int, 2> active_counts{};
    std::array<int, 2> strides{};
    std::size_t pin = 0;

    for (std::size_t dir = 0; dir < 3; ++dir) {
        if (dir == direction)
            continue;

        const BSpline<double>& axis = basis.axis(dir);
        const int line = lines[pin];

        // The last line closes an element, every other one starts it
        const bool last = line == axis.num_elements();
        const int element = last ? line - 1 : line;
        const double parameter = last ? axis.element_end(element)
                                      : axis.element_start(element);

        firsts[pin] = axis.first_active(element);
        active_counts[pin] = axis.num_active();
        strides[pin] = stride[dir];

        axis.eval_on_element(firsts[pin],
                             std::span<const double>(&parameter, 1),
                             weights[pin]);
        ++pin;
    }

    // Contract the net, leaving one control point per function
    const BSpline<double>& axis = basis.axis(direction);
    const int num_functions = axis.num_functions();
    const int curve_stride = stride[direction];

    PointMatrix<double> curve = PointMatrix<double>::Zero(num_functions, 3);
    std::array<int, 2> offset{};
    bool remaining = true;

    while (remaining) {
        int first = 0;
        double weight = 1.;

        for (std::size_t k = 0; k < 2; ++k) {
            first += (firsts[k] + offset[k]) * strides[k];
            weight *= weights[k](offset[k], 0);
        }

        for (int function = 0; function < num_functions; ++function)
            curve.row(function) +=
                weight * coefficients.row(first + function * curve_stride);

        // Advance the active functions in lexicographic order
        remaining = false;

        for (std::size_t k = 0; k < 2; ++k) {
            if (++offset[k] < active_counts[k]) {
                remaining = true;
                break;
            }

            offset[k] = 0;
        }
    }

    return CurvePatch(TensorBSpline<double, 1>(
                          std::array<BSpline<double>, 1>{axis}),
                      std::move(curve));
}

/**
 * @brief Isocurves of a patch along every knot line
 *
 * Each curve lies in the univariate space of the direction it runs along,
 * so that it keeps its degree and knots and reproduces the patch along its
 * line rather than sampling it
 *
 * @param patch Patch the curves are read from
 *
 * @return One group of curves per direction, holding the knot lines of the
 *         remaining directions with the first of them running fastest
 */
std::array<std::vector<CurvePatch>, 3> isocurves(const VolumePatch& patch)
{
    const TensorBSpline<double, 3>& basis = patch.basis();
    std::array<std::vector<CurvePatch>, 3> groups;

    for (std::size_t direction = 0; direction < 3; ++direction) {
        // n elements are bounded by n + 1 knot lines
        std::array<int, 2> counts{};
        std::size_t pin = 0;
        int total = 1;

        for (std::size_t dir = 0; dir < 3; ++dir) {
            if (dir == direction)
                continue;

            counts[pin] = basis.axis(dir).num_elements() + 1;
            total *= counts[pin];
            ++pin;
        }

        groups[direction].reserve(static_cast<std::size_t>(total));

        // Walk the knot lines, the first pinned direction fastest
        std::array<int, 2> lines{};
        bool remaining = true;

        while (remaining) {
            groups[direction].push_back(isocurve(patch, direction, lines));
            remaining = false;

            for (std::size_t k = 0; k < 2; ++k) {
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

} // namespace

void patch(py::module_& module)
{
    py::class_<CurvePatch>(module, "CurvePatch")
        .def_property_readonly("coefficients", &CurvePatch::coefficients);

    py::class_<VolumePatch>(module, "VolumePatch")
        .def(py::init<TensorBSpline<double, 3>, PointMatrix<double>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("coefficients", &VolumePatch::coefficients)
        .def("isocurves", &isocurves);
}

} // namespace iguana::bindings
