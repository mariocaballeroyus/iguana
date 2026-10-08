/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "element_values.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace iguana
{

template<typename Basis, std::size_t n>
ElementValues<Basis, n>::ElementValues(const Patch<Basis, n>& patch,
                                       ValueFlags flags)
    : patch_(patch), flags_(flags)
{
    // J of a curve or surface is not square, so it has no inverse
    if constexpr (dim < n) {
        if (flags.physical_gradients) {
            throw std::invalid_argument("ElementValues: "
                                        "physical gradients need a domain "
                                        "patch");
        }
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::reinit(int element,
                                     const Eigen::MatrixX<Scalar>& points)
{
    using Map = Patch<Basis, n>;
    const Basis& basis = patch_.basis();

    // First active function of each direction, from the element of each
    // direction with the first one running fastest, and the volume of the
    // element in parameter space
    int remaining = element;
    Scalar volume = 1;

    for (std::size_t direction = 0; direction < dim; ++direction) {
        const KnotVector<Scalar>& knots = basis.grid().knots(direction);
        const int axis_element = remaining % knots.num_elements();

        first_active_[direction] =
            knots.element_span(axis_element) - knots.degree();
        volume *= knots.element_end(axis_element)
                  - knots.element_start(axis_element);
        remaining /= knots.num_elements();
    }

    // The points stay for a later shift
    points_ = points;

    basis.active_on_element(element, actives_);
    basis.grad_on_element(first_active_, points, values_, gradients_);

    // The basis that gives the functions also gives the Jacobian of the map
    patch_.tangent_on_element(actives_, gradients_, tangents_);
    Map::measure_on_element(tangents_, measures_);
    sizes_ = (volume * measures_).array().pow(1 / static_cast<Scalar>(dim));

    if constexpr (dim == n) {
        if (flags_.physical_gradients) {
            Map::physical_grad_on_element(tangents_, gradients_,
                                          physical_gradients_);
        }
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::shift(const Eigen::MatrixX<Scalar>& shifts,
                                    int order)
    requires std::same_as<Basis, TensorBSpline<Scalar, dim>>
{
    patch_.basis().taylor_on_element(first_active_, points_, shifts, order,
                                     values_);
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::differentiate(std::size_t direction)
    requires std::same_as<Basis, TensorBSpline<Scalar, dim>>
{
    const int degree = patch_.basis().grid().knots(direction).degree();

    patch_.basis().deriv_on_element(first_active_, points_, direction, degree,
                                    values_);

    // |J^-T e_k| is the measure of a knot line of the direction over that of
    // the patch, by Nanson's formula with the normal e_k
    Eigen::MatrixX<Scalar> normals =
        Eigen::MatrixX<Scalar>::Zero(points_.rows(), dim);
    normals.col(static_cast<Eigen::Index>(direction)).setOnes();

    Eigen::VectorX<Scalar> line_measures;
    Patch<Basis, n>::boundary_measure_on_element(tangents_, normals,
                                                 line_measures);

    const Eigen::VectorX<Scalar> scales =
        (line_measures.array() / measures_.array())
            .pow(static_cast<Scalar>(degree));
    values_ *= scales.asDiagonal();
}

template class ElementValues<TensorBSpline<double, 1>, 2>;
template class ElementValues<TensorBSpline<double, 1>, 3>;
template class ElementValues<TensorBSpline<double, 2>, 2>;
template class ElementValues<TensorBSpline<double, 2>, 3>;
template class ElementValues<TensorBSpline<double, 3>, 3>;
template class ElementValues<TensorNURBS<double, 1>, 2>;
template class ElementValues<TensorNURBS<double, 1>, 3>;
template class ElementValues<TensorNURBS<double, 2>, 2>;
template class ElementValues<TensorNURBS<double, 2>, 3>;
template class ElementValues<TensorNURBS<double, 3>, 3>;

} // namespace iguana
