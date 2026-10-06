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
    // direction with the first one running fastest
    std::array<int, dim> first_active{};
    int remaining = element;

    for (std::size_t direction = 0; direction < dim; ++direction) {
        const KnotVector<Scalar>& knots = basis.grid().knots(direction);
        const int axis_element = remaining % knots.num_elements();

        first_active[direction] =
            knots.element_span(axis_element) - knots.degree();
        remaining /= knots.num_elements();
    }

    basis.active_on_element(element, actives_);
    basis.grad_on_element(first_active, points, values_, gradients_);

    // The basis that gives the functions also gives the Jacobian of the map
    patch_.tangent_on_element(actives_, gradients_, tangents_);
    Map::measure_on_element(tangents_, measures_);

    if constexpr (dim == n) {
        if (flags_.physical_gradients) {
            Map::physical_grad_on_element(tangents_, gradients_,
                                          physical_gradients_);
        }
    }
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
