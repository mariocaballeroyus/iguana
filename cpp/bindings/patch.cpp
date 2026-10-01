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
#include "iguana/utils/multi_index.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using PlanarCurvePatch = Patch<TensorBSpline<double, 1>, 2>;
using PlanarPatch = Patch<TensorBSpline<double, 2>, 2>;
using CurvePatch = Patch<TensorBSpline<double, 1>, 3>;
using SurfacePatch = Patch<TensorBSpline<double, 2>, 3>;
using VolumePatch = Patch<TensorBSpline<double, 3>, 3>;
using NURBSPlanarCurvePatch = Patch<TensorNURBS<double, 1>, 2>;
using NURBSCurvePatch = Patch<TensorNURBS<double, 1>, 3>;
using NURBSSurfacePatch = Patch<TensorNURBS<double, 2>, 3>;

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
 * @brief Coefficients of an isoparametric patch at a knot line of one
 *        direction
 *
 * The coefficients of the functions active at the knot line are contracted
 * against the values of the pinned direction there, so that the result
 * reproduces the patch on the knot line rather than sampling it
 *
 * @param basis Basis of the patch
 * @param coefficients One row per basis function, of any width
 * @param direction Direction pinned at the knot line
 * @param line Knot line of that direction
 *
 * @return One row per function of the directions kept, in their numbering
 *
 * @pre @p line lies in [0,num_elements] of the pinned axis, which
 *      isopatches() ensures
 */
template<std::size_t d>
Eigen::MatrixXd contract(const TensorBSpline<double, d>& basis,
                         const Eigen::MatrixXd& coefficients,
                         std::size_t direction, int line)
{
    static_assert(d > 1, "isopatch: "
                         "a patch needs a direction to keep");

    const BSpline<double>& pinned = basis.axis(direction);

    // The last line closes an element, every other one starts it
    const bool last = line == pinned.knots().num_elements();
    const int element = last ? line - 1 : line;
    const double parameter = last ? pinned.knots().element_end(element)
                                  : pinned.knots().element_start(element);
    const int first = pinned.first_active(element);

    Eigen::MatrixXd values;
    pinned.eval_on_element(first, std::span<const double>(&parameter, 1),
                           values);

    // Function counts of the patch, and of the directions kept
    std::array<int, d> counts{};
    std::array<int, d - 1> kept_counts{};

    for (std::size_t dir = 0, kept = 0; dir < d; ++dir) {
        counts[dir] = basis.axis(dir).num_functions();

        if (dir != direction)
            kept_counts[kept++] = counts[dir];
    }

    const int num_kept = basis.num_functions() / counts[direction];
    Eigen::MatrixXd result =
        Eigen::MatrixXd::Zero(num_kept, coefficients.cols());

    // Every function of the net, the first direction running fastest.
    // Those active at the knot line add to the coefficient that shares
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

            result.row(flatten(kept_index, kept_counts)) +=
                values(offset, 0) * coefficients.row(function);
        }

        ++function;
    } while (next_lexicographic(index, counts));

    return result;
}

/**
 * @brief Isoparametric patch of a B-spline patch at a knot line of one
 *        direction
 *
 * The result keeps the axes of the remaining directions, and its control
 * points are those of the patch contracted at the knot line
 *
 * @param patch Patch the isoparametric patch is read from
 * @param direction Direction pinned at the knot line
 * @param line Knot line of that direction
 *
 * @return Patch of one parametric direction less
 */
template<std::size_t d, std::size_t n>
Patch<TensorBSpline<double, d - 1>, n> isopatch(
    const Patch<TensorBSpline<double, d>, n>& patch, std::size_t direction,
    int line)
{
    const TensorBSpline<double, d>& basis = patch.basis();

    return Patch<TensorBSpline<double, d - 1>, n>(
        TensorBSpline<double, d - 1>(other_axes(
            basis, direction, std::make_index_sequence<d - 1>{})),
        contract(basis, patch.coefficients(), direction, line));
}

/**
 * @brief Isoparametric patch of a NURBS patch at a knot line of one
 *        direction
 *
 * A NURBS patch projects a B-spline patch of one dimension more, whose
 * control points are the weighted ones followed by the weights. Contracting
 * that net and projecting it back gives the isoparametric patch: its weights
 * are the contracted weights, and its control points the contracted
 * weighted ones divided by them
 *
 * @param patch Patch the isoparametric patch is read from
 * @param direction Direction pinned at the knot line
 * @param line Knot line of that direction
 *
 * @return Patch of one parametric direction less
 */
template<std::size_t d, std::size_t n>
Patch<TensorNURBS<double, d - 1>, n> isopatch(
    const Patch<TensorNURBS<double, d>, n>& patch, std::size_t direction,
    int line)
{
    const TensorBSpline<double, d>& bspline = patch.basis().bspline();
    const Eigen::VectorXd& weights = patch.basis().weights();

    Eigen::MatrixXd homogeneous(weights.size(), n + 1);
    homogeneous.leftCols(n) =
        patch.coefficients().array().colwise() * weights.array();
    homogeneous.col(n) = weights;

    const Eigen::MatrixXd kept = contract(bspline, homogeneous, direction,
                                          line);
    const Eigen::VectorXd kept_weights = kept.col(n);
    PointMatrix<double, n> points =
        kept.leftCols(n).array().colwise() / kept_weights.array();

    return Patch<TensorNURBS<double, d - 1>, n>(
        TensorNURBS<double, d - 1>(
            TensorBSpline<double, d - 1>(other_axes(
                bspline, direction, std::make_index_sequence<d - 1>{})),
            kept_weights),
        std::move(points));
}

/**
 * @brief Isoparametric patches of a patch at every knot line
 *
 * @param patch Patch the isoparametric patches are read from
 *
 * @return Patches pinned in each direction in turn, at its knot lines in
 *         increasing order, of the kind isopatch() gives
 */
template<typename Basis, std::size_t n>
auto isopatches(const Patch<Basis, n>& patch)
    -> std::vector<decltype(isopatch(patch, 0, 0))>
{
    std::vector<decltype(isopatch(patch, 0, 0))> result;

    for (std::size_t direction = 0; direction < Basis::dimension;
         ++direction) {
        // n elements are bounded by the knot lines 0 to n
        const int last = patch.basis().grid().knots(direction)
                             .num_elements();

        for (int line = 0; line <= last; ++line)
            result.push_back(isopatch(patch, direction, line));
    }

    return result;
}

} // namespace

void patch(py::module_& module)
{
    py::class_<PlanarCurvePatch>(module, "PlanarCurvePatch")
        .def(py::init<TensorBSpline<double, 1>, PointMatrix<double, 2>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &PlanarCurvePatch::basis)
        .def_property_readonly("coefficients",
                               &PlanarCurvePatch::coefficients);

    py::class_<PlanarPatch>(module, "PlanarPatch")
        .def(py::init<TensorBSpline<double, 2>, PointMatrix<double, 2>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &PlanarPatch::basis)
        .def_property_readonly("coefficients", &PlanarPatch::coefficients)
        .def("isocurves", &isopatches<TensorBSpline<double, 2>, 2>);

    py::class_<CurvePatch>(module, "CurvePatch")
        .def(py::init<TensorBSpline<double, 1>, PointMatrix<double, 3>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &CurvePatch::basis)
        .def_property_readonly("coefficients", &CurvePatch::coefficients);

    py::class_<SurfacePatch>(module, "SurfacePatch")
        .def(py::init<TensorBSpline<double, 2>, PointMatrix<double, 3>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &SurfacePatch::basis)
        .def_property_readonly("coefficients", &SurfacePatch::coefficients)
        .def("isocurves", &isopatches<TensorBSpline<double, 2>, 3>);

    py::class_<VolumePatch>(module, "VolumePatch")
        .def(py::init<TensorBSpline<double, 3>, PointMatrix<double, 3>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &VolumePatch::basis)
        .def_property_readonly("coefficients", &VolumePatch::coefficients)
        .def("isosurfaces", &isopatches<TensorBSpline<double, 3>, 3>);

    py::class_<NURBSPlanarCurvePatch>(module, "NURBSPlanarCurvePatch")
        .def(py::init<TensorNURBS<double, 1>, PointMatrix<double, 2>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &NURBSPlanarCurvePatch::basis)
        .def_property_readonly("coefficients",
                               &NURBSPlanarCurvePatch::coefficients);

    py::class_<NURBSCurvePatch>(module, "NURBSCurvePatch")
        .def(py::init<TensorNURBS<double, 1>, PointMatrix<double, 3>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &NURBSCurvePatch::basis)
        .def_property_readonly("coefficients",
                               &NURBSCurvePatch::coefficients);

    py::class_<NURBSSurfacePatch>(module, "NURBSSurfacePatch")
        .def(py::init<TensorNURBS<double, 2>, PointMatrix<double, 3>>(),
             py::arg("basis"), py::arg("coefficients"))
        .def_property_readonly("basis", &NURBSSurfacePatch::basis)
        .def_property_readonly("coefficients",
                               &NURBSSurfacePatch::coefficients)
        .def("isocurves", &isopatches<TensorNURBS<double, 2>, 3>);
}

} // namespace iguana::bindings
