/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <stdexcept>
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

using SurfaceQuadrature = DomainQuadrature<double, 2>;
using VolumeQuadrature = DomainQuadrature<double, 3>;

/// @brief Fills the cells of one type with a Gauss-Legendre rule, which
///        Python never handles itself
template<std::size_t d>
void fill_gauss_legendre(DomainQuadrature<double, d>& quadrature,
                         const TensorDomain<double, d>& domain,
                         CellType cell_type,
                         const std::array<int, d>& num_points)
{
    quadrature.fill(domain, cell_type, GaussLegendre<double, d>(num_points));
}

/**
 * @brief Positions of the points in physical space, the only form Python
 *        needs, as it draws them
 *
 * @throws std::invalid_argument If an element of the quadrature lies
 *         outside the domain
 */
template<std::size_t d>
PointMatrix<double> positions(const DomainQuadrature<double, d>& quadrature,
                              const TensorDomain<double, d>& domain)
{
    const Eigen::VectorXi& offsets = quadrature.offsets();

    // Position of each element among the held ones, or -1 if not held
    std::vector<int> held(domain.num_elements(), -1);

    for (int position = 0; position < quadrature.num_elements(); ++position) {
        const int element = quadrature.elements()(position);

        if (element < 0 || element >= domain.num_elements())
            throw std::invalid_argument("DomainQuadrature: "
                                        "the elements must lie in the "
                                        "domain");

        held[element] = position;
    }

    PointMatrix<double> result(quadrature.num_points(), 3);

    // Buffers reused over the elements
    Eigen::MatrixXd parameters;
    Eigen::MatrixXd values;
    Eigen::VectorXi actives;
    PointMatrix<double> element_positions;

    for (const TensorDomainIterator<double, d>& element : domain) {
        const int position = held[element.index()];

        if (position < 0)
            continue;

        const int first = offsets(position);
        const int count = offsets(position + 1) - first;

        parameters = quadrature.points().middleRows(first, count);
        domain.basis().eval_on_element(element.first_active(), parameters,
                                       values);
        domain.basis().active_on_element(element.index(), actives);
        domain.patch().position_on_element(actives, values,
                                           element_positions);

        result.middleRows(first, count) = element_positions;
    }

    return result;
}

} // namespace

void quadrature(py::module_& module)
{
    py::class_<SurfaceQuadrature>(module, "SurfaceQuadrature")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<2>,
             py::arg("domain"), py::arg("cell_type"), py::arg("num_points"))
        .def_property_readonly("num_elements",
                               &SurfaceQuadrature::num_elements)
        .def_property_readonly("num_points", &SurfaceQuadrature::num_points)
        .def("positions", &positions<2>, py::arg("domain"));

    py::class_<VolumeQuadrature>(module, "VolumeQuadrature")
        .def(py::init<>())
        .def("fill_gauss_legendre", &fill_gauss_legendre<3>,
             py::arg("domain"), py::arg("cell_type"), py::arg("num_points"))
        .def_property_readonly("num_elements",
                               &VolumeQuadrature::num_elements)
        .def_property_readonly("num_points", &VolumeQuadrature::num_points)
        .def("positions", &positions<3>, py::arg("domain"));
}

} // namespace iguana::bindings
