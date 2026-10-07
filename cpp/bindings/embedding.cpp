/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "iguana/iguana.hpp"

namespace py = pybind11;

namespace iguana::bindings
{

namespace
{

using EmbeddedDomain2d = EmbeddedDomain<double, 2>;
using EmbeddedDomain3d = EmbeddedDomain<double, 3>;
using EmbeddedBoundary2d = EmbeddedBoundary<double, 2>;
using SurrogateBoundary2d = SurrogateBoundary<double, 2>;
using PlanarPatch = Patch<TensorBSpline<double, 2>, 2>;

/// @brief Element of the grid holding each piece, in the order of the
///        pieces
Eigen::VectorXi elements(const EmbeddedBoundary2d& boundary)
{
    Eigen::VectorXi result(boundary.num_pieces());
    int piece = 0;

    for (int element = 0; element < boundary.num_elements(); ++element) {
        const int count =
            static_cast<int>(boundary.pieces_on_element(element).size());

        result.segment(piece, count).setConstant(element);
        piece += count;
    }

    return result;
}

/// @brief Face of each piece
Eigen::VectorXi faces(const EmbeddedBoundary2d& boundary)
{
    Eigen::VectorXi result(boundary.num_pieces());
    int piece = 0;

    for (int element = 0; element < boundary.num_elements(); ++element)
        for (const EmbeddedBoundary2d::Piece& entry :
             boundary.pieces_on_element(element))
            result(piece++) = entry.face;

    return result;
}

/// @brief Parameters of its face at which each piece starts and ends, one
///        row per piece
Eigen::MatrixXd intervals(const EmbeddedBoundary2d& boundary)
{
    Eigen::MatrixXd result(boundary.num_pieces(), 2);
    int piece = 0;

    for (int element = 0; element < boundary.num_elements(); ++element)
        for (const EmbeddedBoundary2d::Piece& entry :
             boundary.pieces_on_element(element))
            result.row(piece++) << entry.start, entry.end;

    return result;
}

/// @brief Surrogate boundary of the inside cells among the elements of a
///        patch
SurrogateBoundary2d surrogate_boundary(const PlanarPatch& patch,
                                       const EmbeddedDomain2d& domain)
{
    return {patch.basis().grid(), domain};
}

/// @brief Element of the grid holding each face, in the order of the faces
Eigen::VectorXi face_elements(const SurrogateBoundary2d& boundary)
{
    Eigen::VectorXi result(boundary.num_faces());
    int face = 0;

    for (int element = 0; element < boundary.num_elements(); ++element) {
        const int count =
            static_cast<int>(boundary.faces_on_element(element).size());

        result.segment(face, count).setConstant(element);
        face += count;
    }

    return result;
}

} // namespace

void embedding(py::module_& module)
{
    py::enum_<CellType>(module, "CellType")
        .value("outside", CellType::outside)
        .value("inside", CellType::inside)
        .value("cut", CellType::cut);

    py::class_<EmbeddedDomain2d>(module, "EmbeddedDomain2d")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));

    py::class_<EmbeddedDomain3d>(module, "EmbeddedDomain3d")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));

    py::class_<EmbeddedBoundary2d>(module, "EmbeddedBoundary2d")
        .def(py::init<const PlanarPatch&, const Boundary<double, 2>&>(),
             py::arg("patch"), py::arg("boundary"))
        .def_property_readonly("num_pieces", &EmbeddedBoundary2d::num_pieces)
        .def_property_readonly("elements", &elements)
        .def_property_readonly("faces", &faces)
        .def_property_readonly("intervals", &intervals);

    py::class_<SurrogateBoundary2d>(module, "SurrogateBoundary2d")
        .def(py::init(&surrogate_boundary), py::arg("patch"),
             py::arg("domain"))
        .def_property_readonly("num_faces", &SurrogateBoundary2d::num_faces)
        .def_property_readonly("elements", &face_elements);
}

} // namespace iguana::bindings
