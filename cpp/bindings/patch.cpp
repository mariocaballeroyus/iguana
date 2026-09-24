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
#include "iguana/multi_index.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using CurvePatch = Patch<double, 1>;
using SurfacePatch = Patch<double, 2>;
using VolumePatch = Patch<double, 3>;

/**
 * @brief Axes of a basis, leaving out the one of a given direction
 *
 * @param basis Basis the axes are read from
 * @param direction Direction whose axis is left out
 *
 * @return Remaining axes, in increasing order of direction
 */
template<std::size_t d, std::size_t... index>
std::array<BSpline<double>, d - 1> other_axes(
    const TensorBSpline<double, d>& basis, std::size_t direction,
    std::index_sequence<index...>)
{
    return {basis.axis(index < direction ? index : index + 1)...};
}

/**
 * @brief Isoparametric patch at a knot line of one direction
 *
 * The result keeps the axes of the remaining directions, so that it
 * reproduces the patch on the knot line rather than sampling it. Only its
 * control points are formed, by contracting the control net against the
 * values of the pinned direction at the knot line
 *
 * @param patch Patch the isoparametric patch is read from
 * @param direction Direction pinned at the knot line
 * @param line Knot line of that direction
 *
 * @return Patch of one parametric direction less
 *
 * @pre @p line lies in [0,num_elements] of the pinned axis, which
 *      isopatches() ensures
 */
template<std::size_t d>
Patch<double, d - 1> isopatch(const Patch<double, d>& patch,
                              std::size_t direction,
                              int line)
{
    static_assert(d > 1, "isopatch: "
                         "a patch needs a direction to keep");

    const TensorBSpline<double, d>& basis = patch.basis();
    const BSpline<double>& pinned = basis.axis(direction);

    // The last line closes an element, every other one starts it
    const bool last = line == pinned.num_elements();
    const int element = last ? line - 1 : line;
    const double parameter = last ? pinned.element_end(element)
                                  : pinned.element_start(element);
    const int first = pinned.first_active(element);

    Eigen::MatrixXd weights;
    pinned.eval_on_element(first, std::span<const double>(&parameter, 1),
                           weights);

    // Function counts of the patch, and of the directions kept
    std::array<int, d> counts{};
    std::array<int, d - 1> kept_counts{};

    for (std::size_t dir = 0, kept = 0; dir < d; ++dir) {
        counts[dir] = basis.axis(dir).num_functions();

        if (dir != direction)
            kept_counts[kept++] = counts[dir];
    }

    const int num_kept = basis.num_functions() / counts[direction];
    PointMatrix<double> coefficients =
        PointMatrix<double>::Zero(num_kept, 3);

    // Every function of the net, the first direction running fastest.
    // Those active at the knot line add to the control point that shares
    // their indices in the directions kept
    std::array<int, d> index{};
    int function = 0;

    do {
        const int offset = index[direction] - first;

        if (offset >= 0 && offset < pinned.num_active()) {
            std::array<int, d - 1> kept_index{};

            for (std::size_t dir = 0, kept = 0; dir < d; ++dir) {
                if (dir != direction)
                    kept_index[kept++] = index[dir];
            }

            coefficients.row(flatten(kept_index, kept_counts)) +=
                weights(offset, 0) * patch.coefficients().row(function);
        }

        ++function;
    } while (next_lexicographic(index, counts));

    return Patch<double, d - 1>(
        TensorBSpline<double, d - 1>(other_axes(
            basis, direction, std::make_index_sequence<d - 1>{})),
        std::move(coefficients));
}

/**
 * @brief Isoparametric patches of a patch at every knot line
 *
 * @param patch Patch the isoparametric patches are read from
 *
 * @return Patches pinned in each direction in turn, at its knot lines in
 *         increasing order
 */
template<std::size_t d>
std::vector<Patch<double, d - 1>> isopatches(const Patch<double, d>& patch)
{
    std::vector<Patch<double, d - 1>> result;

    for (std::size_t direction = 0; direction < d; ++direction) {
        // n elements are bounded by the knot lines 0 to n
        const int last = patch.basis().axis(direction).num_elements();

        for (int line = 0; line <= last; ++line)
            result.push_back(isopatch(patch, direction, line));
    }

    return result;
}

} // namespace

void patch(py::module_& module)
{
    py::class_<CurvePatch>(module, "CurvePatch")
        .def_property_readonly("basis", &CurvePatch::basis)
        .def_property_readonly("coefficients", &CurvePatch::coefficients);

    py::class_<SurfacePatch>(module, "SurfacePatch")
        .def_property_readonly("basis", &SurfacePatch::basis)
        .def_property_readonly("coefficients", &SurfacePatch::coefficients)
        .def("isocurves", &isopatches<2>);

    py::class_<VolumePatch>(module, "VolumePatch")
        .def(py::init<TensorBSpline<double, 3>, PointMatrix<double>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &VolumePatch::basis)
        .def_property_readonly("coefficients", &VolumePatch::coefficients)
        .def("isosurfaces", &isopatches<3>);
}

} // namespace iguana::bindings
