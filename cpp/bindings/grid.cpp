/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
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

using HierarchicalGrid2d = HierarchicalGrid<double, 2>;
using HierarchicalGrid3d = HierarchicalGrid<double, 3>;

/// @brief Knot vector of each direction, from its degree and knots
template<std::size_t d, std::size_t... direction>
std::array<KnotVector<double>, d> knot_vectors(
    const std::array<int, d>& degrees,
    const std::array<std::vector<double>, d>& knots,
    std::index_sequence<direction...>)
{
    return {KnotVector<double>(degrees[direction], knots[direction])...};
}

/// @brief Grid on the knot vector of each direction, with every element
///        active on level 0
template<std::size_t d>
HierarchicalGrid<double, d> create(
    const std::array<int, d>& degrees,
    const std::array<std::vector<double>, d>& knots)
{
    return HierarchicalGrid<double, d>(TensorGrid<double, d>(
        knot_vectors(degrees, knots, std::make_index_sequence<d>{})));
}

/// @brief Grid with the marked elements replaced by their children
template<std::size_t d>
HierarchicalGrid<double, d> refined(
    const HierarchicalGrid<double, d>& grid,
    const std::vector<int>& elements)
{
    return refine(grid, elements);
}

/// @brief Level of each active element
template<std::size_t d>
Eigen::VectorXi levels(const HierarchicalGrid<double, d>& grid)
{
    Eigen::VectorXi result(grid.num_elements());

    for (const HierarchicalGridIterator<double, d>& element : grid)
        result(element.index()) = element.level();

    return result;
}

/// @brief Parameters at which each active element starts, followed by those
///        at which it ends, with size (num_elements, 2d)
template<std::size_t d>
Eigen::MatrixXd bounds(const HierarchicalGrid<double, d>& grid)
{
    Eigen::MatrixXd result(grid.num_elements(), 2 * d);

    for (const HierarchicalGridIterator<double, d>& element : grid) {
        for (std::size_t direction = 0; direction < d; ++direction) {
            result(element.index(), direction) = element.start()[direction];
            result(element.index(), d + direction) = element.end()[direction];
        }
    }

    return result;
}

} // namespace

void grid(py::module_& module)
{
    py::class_<HierarchicalGrid2d>(module, "HierarchicalGrid2d")
        .def(py::init(&create<2>), py::arg("degrees"), py::arg("knots"))
        .def("refine", &refined<2>, py::arg("elements"))
        .def_property_readonly("num_levels",
                               &HierarchicalGrid2d::num_levels)
        .def_property_readonly("num_elements",
                               &HierarchicalGrid2d::num_elements)
        .def_property_readonly("levels", &levels<2>)
        .def_property_readonly("bounds", &bounds<2>);

    py::class_<HierarchicalGrid3d>(module, "HierarchicalGrid3d")
        .def(py::init(&create<3>), py::arg("degrees"), py::arg("knots"))
        .def("refine", &refined<3>, py::arg("elements"))
        .def_property_readonly("num_levels",
                               &HierarchicalGrid3d::num_levels)
        .def_property_readonly("num_elements",
                               &HierarchicalGrid3d::num_elements)
        .def_property_readonly("levels", &levels<3>)
        .def_property_readonly("bounds", &bounds<3>);
}

} // namespace iguana::bindings
