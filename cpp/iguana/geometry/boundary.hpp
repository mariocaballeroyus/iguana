/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_BOUNDARY_HPP
#define IGUANA_GEOMETRY_BOUNDARY_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Boundary of a physical domain, made of NURBS faces in physical
 *        space
 *
 * Each face is a patch of one parametric direction fewer than the space, a
 * curve in the plane or a surface in space.The sign of each face tells 
 * whether that normal points out of the domain, so that the faces need not be 
 * parametrized alike.
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

private:
    /// @brief Faces of the boundary
    std::vector<Face> faces_;

    /// @brief Sign of each face, 1 or -1
    std::vector<int> signs_;
};

} // namespace iguana

#endif // IGUANA_GEOMETRY_BOUNDARY_HPP
