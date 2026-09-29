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

using SurfaceHierarchicalDomain = HierarchicalDomain<double, 2>;
using VolumeHierarchicalDomain = HierarchicalDomain<double, 3>;

/// @brief Knot vector of each direction, from its degree and knots
template<std::size_t d, std::size_t... direction>
std::array<KnotVector<double>, d> knot_vectors(
    const std::array<int, d>& degrees,
    const std::array<std::vector<double>, d>& knots,
    std::index_sequence<direction...>)
{
    return {KnotVector<double>(degrees[direction], knots[direction])...};
}

/// @brief Domain on the knot vector of each direction, with every element
///        active on level 0
template<std::size_t d>
HierarchicalDomain<double, d> create(
    const std::array<int, d>& degrees,
    const std::array<std::vector<double>, d>& knots)
{
    return HierarchicalDomain<double, d>(TensorDomain<double, d>(
        knot_vectors(degrees, knots, std::make_index_sequence<d>{})));
}

/// @brief Domain with the marked elements replaced by their children
template<std::size_t d>
HierarchicalDomain<double, d> refined(
    const HierarchicalDomain<double, d>& domain,
    const std::vector<int>& elements)
{
    return refine(domain, elements);
}

/// @brief Level of each active element
template<std::size_t d>
Eigen::VectorXi levels(const HierarchicalDomain<double, d>& domain)
{
    Eigen::VectorXi result(domain.num_elements());

    for (const HierarchicalDomainIterator<double, d>& element : domain)
        result(element.index()) = element.level();

    return result;
}

/// @brief Parameters at which each active element starts, followed by those
///        at which it ends, with size (num_elements, 2d)
template<std::size_t d>
Eigen::MatrixXd bounds(const HierarchicalDomain<double, d>& domain)
{
    Eigen::MatrixXd result(domain.num_elements(), 2 * d);

    for (const HierarchicalDomainIterator<double, d>& element : domain) {
        for (std::size_t direction = 0; direction < d; ++direction) {
            result(element.index(), direction) = element.start()[direction];
            result(element.index(), d + direction) = element.end()[direction];
        }
    }

    return result;
}

} // namespace

void domain(py::module_& module)
{
    py::class_<SurfaceHierarchicalDomain>(module, "SurfaceHierarchicalDomain")
        .def(py::init(&create<2>), py::arg("degrees"), py::arg("knots"))
        .def("refine", &refined<2>, py::arg("elements"))
        .def_property_readonly("num_levels",
                               &SurfaceHierarchicalDomain::num_levels)
        .def_property_readonly("num_elements",
                               &SurfaceHierarchicalDomain::num_elements)
        .def_property_readonly("levels", &levels<2>)
        .def_property_readonly("bounds", &bounds<2>);

    py::class_<VolumeHierarchicalDomain>(module, "VolumeHierarchicalDomain")
        .def(py::init(&create<3>), py::arg("degrees"), py::arg("knots"))
        .def("refine", &refined<3>, py::arg("elements"))
        .def_property_readonly("num_levels",
                               &VolumeHierarchicalDomain::num_levels)
        .def_property_readonly("num_elements",
                               &VolumeHierarchicalDomain::num_elements)
        .def_property_readonly("levels", &levels<3>)
        .def_property_readonly("bounds", &bounds<3>);
}

} // namespace iguana::bindings
