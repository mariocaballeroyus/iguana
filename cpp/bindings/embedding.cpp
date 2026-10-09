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

using CellClassification2d = CellClassification<double, 2>;
using CellClassification3d = CellClassification<double, 3>;
using EmbeddedBoundary2d = EmbeddedBoundary<double, 2>;
using SurrogateBoundary2d = SurrogateBoundary<double, 2>;
using GhostFaces2d = GhostFaces<double, 2>;
using JumpFaces2d = JumpFaces<double, 2>;
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
SurrogateBoundary2d surrogate_boundary(
    const PlanarPatch& patch, const CellClassification2d& classification)
{
    return {patch.basis().grid(), classification};
}

/// @brief Faces where a ghost penalty acts among the elements of a patch
GhostFaces2d ghost_faces(const PlanarPatch& patch,
                         const CellClassification2d& classification)
{
    return {patch.basis().grid(), classification};
}

/// @brief Faces across which the volume fraction may jump among the
///        elements of a patch
JumpFaces2d jump_faces(const PlanarPatch& patch,
                       const CellClassification2d& classification)
{
    return {patch.basis().grid(), classification};
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

    py::class_<CellClassification2d>(module, "CellClassification2d")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));

    py::class_<CellClassification3d>(module, "CellClassification3d")
        .def(py::init<std::vector<CellType>>(), py::arg("cell_types"));

    // The volume fractions of the elements of a planar patch inside a
    // boundary in its plane, and the cell types they give
    module.def("volume_fractions",
               &volume_fractions<TensorBSpline<double, 2>>, py::arg("patch"),
               py::arg("boundary"));
    module.def("cell_types", &cell_types<double>,
               py::arg("volume_fractions"));

    py::class_<EmbeddedBoundary2d>(module, "EmbeddedBoundary2d")
        .def(py::init<const PlanarPatch&, const Boundary<double, 2>&>(),
             py::arg("patch"), py::arg("boundary"))
        .def_property_readonly("num_pieces", &EmbeddedBoundary2d::num_pieces)
        .def_property_readonly("elements", &elements)
        .def_property_readonly("faces", &faces)
        .def_property_readonly("intervals", &intervals);

    py::class_<SurrogateBoundary2d>(module, "SurrogateBoundary2d")
        .def(py::init(&surrogate_boundary), py::arg("patch"),
             py::arg("classification"))
        .def_property_readonly("num_faces", &SurrogateBoundary2d::num_faces)
        .def_property_readonly("elements", &face_elements);

    py::class_<GhostFaces2d>(module, "GhostFaces2d")
        .def(py::init(&ghost_faces), py::arg("patch"),
             py::arg("classification"))
        .def_property_readonly("num_faces", &GhostFaces2d::num_faces);

    py::class_<JumpFaces2d>(module, "JumpFaces2d")
        .def(py::init(&jump_faces), py::arg("patch"),
             py::arg("classification"))
        .def_property_readonly("num_faces", &JumpFaces2d::num_faces);
}

} // namespace iguana::bindings
