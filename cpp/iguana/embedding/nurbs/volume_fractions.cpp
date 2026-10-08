/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "volume_fractions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "iguana/embedding/embedded_boundary.hpp"
#include "iguana/quadrature/boundary_quadrature.hpp"
#include "iguana/quadrature/gauss_legendre/gauss_legendre.hpp"

namespace iguana
{

template<typename Basis>
    requires (Basis::dimension == 2)
Eigen::VectorX<typename Basis::Scalar> volume_fractions(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary)
{
    using T = typename Basis::Scalar;

    const EmbeddedBoundary<T, 2> embedded(patch, boundary);
    const BoundaryQuadrature<T, 2> quadrature(embedded,
                                              GaussLegendre<T, 1>(8));

    const KnotVector<T>& along_x = patch.basis().grid().knots(0);
    const KnotVector<T>& along_y = patch.basis().grid().knots(1);
    const int columns = along_x.num_elements();
    const int num_elements = columns * along_y.num_elements();

    // Integrals of (x - x0) n_x ds and n_x ds over the pieces in each cell,
    // and of x n_x ds over all of them, the area the boundary encloses
    Eigen::VectorX<T> first = Eigen::VectorX<T>::Zero(num_elements);
    Eigen::VectorX<T> flux = Eigen::VectorX<T>::Zero(num_elements);
    std::vector<bool> held(static_cast<std::size_t>(num_elements), false);
    T area = 0;

    for (int entry = 0; entry < quadrature.num_elements(); ++entry) {
        const int cell = quadrature.elements()(entry);
        const T start = along_x.element_start(cell % columns);

        held[static_cast<std::size_t>(cell)] = true;

        for (int point = quadrature.offsets()(entry);
             point < quadrature.offsets()(entry + 1); ++point) {
            const T x = quadrature.points()(point, 0);
            const T measure =
                quadrature.weights()(point) * quadrature.normals()(point, 0);

            first(cell) += (x - start) * measure;
            flux(cell) += measure;
            area += x * measure;
        }
    }

    const T tolerance = EmbeddedBoundary<T, 2>::tolerance;
    Eigen::VectorX<T> fractions(num_elements);

    for (int row = 0; row < along_y.num_elements(); ++row) {
        const T height = along_y.element_end(row) - along_y.element_start(row);

        // Flux of the pieces to the right of the cell, and from infinity
        // for an unbounded domain, running leftwards along the row
        T right = area < 0 ? height : T{0};

        for (int column = columns - 1; column >= 0; --column) {
            const int cell = column + columns * row;
            const T width =
                along_x.element_end(column) - along_x.element_start(column);

            T fraction = std::clamp((first(cell) + width * right)
                                        / (width * height),
                                    T{0}, T{1});

            if (!held[static_cast<std::size_t>(cell)])
                fraction = std::round(fraction);
            else if (fraction < tolerance)
                fraction = 0;
            else if (fraction > 1 - tolerance)
                fraction = 1;

            fractions(cell) = fraction;
            right += flux(cell);
        }
    }

    return fractions;
}

template Eigen::VectorX<double> volume_fractions(
    const Patch<TensorBSpline<double, 2>, 2>&, const Boundary<double, 2>&);
template Eigen::VectorX<double> volume_fractions(
    const Patch<TensorNURBS<double, 2>, 2>&, const Boundary<double, 2>&);

} // namespace iguana
