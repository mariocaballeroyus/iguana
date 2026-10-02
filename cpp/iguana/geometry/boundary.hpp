/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_BOUNDARY_HPP
#define IGUANA_GEOMETRY_BOUNDARY_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Boundary of a physical domain, made of NURBS faces in physical
 *        space
 *
 * Each face is a patch of one parametric direction fewer than the space, a
 * curve in the plane or a surface in space. Its parametrization gives it a
 * normal, and the sign of each face tells whether that normal points out of
 * the domain, so that the faces need not be parametrized alike
 *
 * @tparam T Floating-point type
 * @tparam n Dimension of the physical space, two or three
 */
template<std::floating_point T, std::size_t n>
class Boundary
{
    static_assert(n == 2 || n == 3,
                  "Boundary: "
                  "the space must have two or three dimensions");

public:
    /// @brief Face, a NURBS patch of one direction fewer than the space
    using Face = Patch<TensorNURBS<T, n - 1>, n>;

    /**
     * @brief Constructs the boundary from its faces and their orientation
     *
     * @param faces Faces of the boundary
     * @param signs Sign of each face, 1 where its normal points out of the
     *        domain and -1 where it points into it
     *
     * @throws std::invalid_argument If there is not one sign per face, or
     *         if a sign is neither 1 nor -1
     */
    Boundary(std::vector<Face> faces, std::vector<int> signs);

    /// @brief Number of faces
    constexpr int num_faces() const noexcept
    { return static_cast<int>(faces_.size()); }

    /**
     * @brief Face of the boundary
     *
     * @param index Face index
     *
     * @pre @p index lies in [0, num_faces())
     */
    constexpr const Face& face(int index) const noexcept
    { return faces_[static_cast<std::size_t>(index)]; }

    /**
     * @brief Sign of a face, 1 where its normal points out of the domain and
     *        -1 where it points into it
     *
     * @param index Face index
     *
     * @pre @p index lies in [0, num_faces())
     */
    constexpr int sign(int index) const noexcept
    { return signs_[static_cast<std::size_t>(index)]; }

    /**
     * @brief Unit normals of a face at points of one of its elements,
     *        pointing out of the domain
     *
     * The normal of the parametrization, (y', -x') along a curve and
     * x_u x x_v on a surface, is normalized and takes the sign of the face
     *
     * @param face Face index
     * @param actives Functions of the face that are non-zero on the element,
     *        as given by active_on_element() of its basis
     * @param gradients Their derivatives along each direction of the face at
     *        the points, each of size (num_active, num_points), as given by
     *        grad_on_element() of its basis
     * @param normals Output buffer of size (num_points, n), resized if its
     *        shape changes
     *
     * @pre @p face lies in [0, num_faces()), @p actives and @p gradients come
     *      from the same element, and the tangents of the face span a line
     *      or a plane at every point
     */
    void normal_on_element(
        int face, const Eigen::VectorXi& actives,
        const std::array<Eigen::MatrixX<T>, n - 1>& gradients,
        PointMatrix<T, n>& normals) const;

private:
    /// @brief Faces of the boundary
    std::vector<Face> faces_;

    /// @brief Sign of each face, 1 or -1
    std::vector<int> signs_;
};

} // namespace iguana

#endif // IGUANA_GEOMETRY_BOUNDARY_HPP
