/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_GEOMETRY_DIFFERENTIAL_HPP
#define IGUANA_GEOMETRY_DIFFERENTIAL_HPP

#include<array>
#include<concepts>
#include<cstddef>

#include<Eigen/Core>

namespace iguana
{

/**
 * @brief The metric of a map at points, computed from its tangents.
 *
 * The metric is the first fundamental form \f$ g_{ij} = g_i \cdot g_j
 * \f$. Being symmetric, it is stored as one column per distinct
 * component and one row per point. The diagonal components first in
 * direction order, then the mixed pairs in the lexicographic order.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the fields.
 *
 * @param tangents The tangent of every direction at the points, as
 *        given by tangents().
 * @param g Output buffer of size (num_points,d(d+1)/2), resized if its
 *        shape changes.
 */
template<std::size_t d, std::floating_point T>
void metric(const std::array<Eigen::MatrixX3<T>, d>& tangents,
            Eigen::MatrixX<T>& g);

/**
 * @brief The measure of a map at points, computed from its metric.
 *
 * The measure is \f$ \sqrt{\det g} \f$, the factor a quadrature weight
 * takes to integrate over the image of the map: the Jacobian
 * determinant of a volume, the area element of a surface and the line
 * element of a curve.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the fields.
 *
 * @param g The metric components at the points, as given by metric().
 * @param measures Output buffer of size (num_points,), resized if its
 *        size changes.
 */
template<std::size_t d, std::floating_point T>
void measure(const Eigen::MatrixX<T>& g,
             Eigen::VectorX<T>& measures);

/**
 * @brief The inverse of a metric at points, the contravariant metric.
 *
 * The inverse $ g^{ij}$ raises indices: it pushes parametric gradients 
 * forward through the map. It is stored as the metric is, the diagonal
 * first and then the mixed pairs.
 *
 * @tparam d Number of parametric directions. It is not deduced.
 * @tparam T Floating-point type of the fields.
 *
 * @param g The metric components at the points, as given by metric().
 * @param inverse Output buffer of size (num_points,d(d+1)/2), resized
 *        if its shape changes.
 *
 * @pre The metric is invertible at every point, as it is wherever the
 *      tangents are independent. It is not checked.
 */
template<std::size_t d, std::floating_point T>
void metric_inverse(const Eigen::MatrixX<T>& g,
                    Eigen::MatrixX<T>& inverse);

} // namespace iguana

#endif // IGUANA_GEOMETRY_DIFFERENTIAL_HPP
