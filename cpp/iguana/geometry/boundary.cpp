/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "boundary.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <Eigen/Geometry>

namespace iguana
{

template<std::floating_point T, std::size_t n>
Boundary<T, n>::Boundary(std::vector<Face> faces, std::vector<int> signs)
    : faces_(std::move(faces)),
      signs_(std::move(signs))
{
    if (signs_.size() != faces_.size())
        throw std::invalid_argument("Boundary: "
                                    "there must be one sign per face");

    if (!std::ranges::all_of(signs_, [](int sign) {
            return sign == 1 || sign == -1;
        }))
        throw std::invalid_argument("Boundary: "
                                    "the signs must be 1 or -1");
}

template<std::floating_point T, std::size_t n>
void Boundary<T, n>::normal_on_element(
    int face, const Eigen::VectorXi& actives,
    const std::array<Eigen::MatrixX<T>, n - 1>& gradients,
    PointMatrix<T, n>& normals) const
{
    const std::size_t index = static_cast<std::size_t>(face);

    std::array<PointMatrix<T, n>, n - 1> tangents;
    faces_[index].tangent_on_element(actives, gradients, tangents);

    normals.resize(tangents[0].rows(), n);

    if constexpr (n == 2) {
        // The tangent turned clockwise, (y', -x')
        normals.col(0) = tangents[0].col(1);
        normals.col(1) = -tangents[0].col(0);
    }
    else if constexpr (n == 3) {
        for (Eigen::Index point = 0; point < normals.rows(); ++point)
            normals.row(point) =
                tangents[0].row(point).cross(tangents[1].row(point));
    }

    normals.rowwise().normalize();
    normals *= static_cast<T>(signs_[index]);
}

template class Boundary<double, 2>;
template class Boundary<double, 3>;

} // namespace iguana
