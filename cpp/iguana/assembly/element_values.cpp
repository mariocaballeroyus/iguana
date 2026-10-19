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

namespace
{

/// @brief First active function of each direction of a basis on an element,
///        from the element of each direction with the first one running
///        fastest
template<typename Basis>
std::array<int, Basis::dimension> first_active_on(const Basis& basis,
                                                  int element)
{
    std::array<int, Basis::dimension> first_active{};
    int remaining = element;

    for (std::size_t direction = 0; direction < Basis::dimension;
         ++direction) {
        const KnotVector<typename Basis::Scalar>& knots =
            basis.grid().knots(direction);
        const int axis_element = remaining % knots.num_elements();

        first_active[direction] =
            knots.element_span(axis_element) - knots.degree();
        remaining /= knots.num_elements();
    }

    return first_active;
}

} // namespace

template<typename Basis, std::size_t n>
ElementValues<Basis, n>::ElementValues(const Patch<Basis, n>& patch,
                                       ValueFlags flags)
    : ElementValues(patch, patch.basis(), flags)
{
}

template<typename Basis, std::size_t n>
ElementValues<Basis, n>::ElementValues(const Patch<Basis, n>& patch,
                                       const Basis& basis, ValueFlags flags)
    : patch_(patch), basis_(basis),
      isoparametric_(basis == patch.basis()), flags_(flags)
{
    // One element index locates the points in both bases, which share their
    // knot lines
    if (basis.grid().lines() != patch.basis().grid().lines())
        throw std::invalid_argument("ElementValues: "
                                    "the basis must have the elements of the "
                                    "patch");

    // J of a curve or surface is not square, so it has no inverse
    if constexpr (dim < n) {
        if (flags.physical_gradients || flags.physical_hessians) {
            throw std::invalid_argument("ElementValues: "
                                        "physical derivatives need a domain "
                                        "patch");
        }
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::reinit(int element,
                                     const Eigen::MatrixX<Scalar>& points)
{
    using Map = Patch<Basis, n>;

    // Volume of the element in parameter space, the same in both bases
    int remaining = element;
    Scalar volume = 1;

    for (std::size_t direction = 0; direction < dim; ++direction) {
        const KnotVector<Scalar>& knots = basis_.grid().knots(direction);
        const int axis_element = remaining % knots.num_elements();

        volume *= knots.element_end(axis_element)
                  - knots.element_start(axis_element);
        remaining /= knots.num_elements();
    }

    // The points stay for a later shift
    points_ = points;
    first_active_ = first_active_on(basis_, element);
    basis_.active_on_element(element, actives_);

    // One evaluation gives the second derivatives with the first
    if (flags_.physical_hessians)
        basis_.hess_on_element(first_active_, points, values_, gradients_,
                               hessians_);
    else
        basis_.grad_on_element(first_active_, points, values_, gradients_);

    // The patch's own functions give the map, which are those just
    // evaluated when the basis is the patch's
    if (isoparametric_) {
        patch_.tangent_on_element(actives_, gradients_, tangents_);

        if (flags_.physical_hessians)
            patch_.second_tangent_on_element(actives_, hessians_,
                                             second_tangents_);
    } else {
        const Basis& geometry = patch_.basis();
        const std::array<int, dim> map_first_active =
            first_active_on(geometry, element);
        geometry.active_on_element(element, map_actives_);

        if (flags_.physical_hessians) {
            geometry.hess_on_element(map_first_active, points, map_values_,
                                     map_gradients_, map_hessians_);
            patch_.second_tangent_on_element(map_actives_, map_hessians_,
                                             second_tangents_);
        } else {
            geometry.grad_on_element(map_first_active, points, map_values_,
                                     map_gradients_);
        }

        patch_.tangent_on_element(map_actives_, map_gradients_, tangents_);
    }

    Map::measure_on_element(tangents_, measures_);
    sizes_ = (volume * measures_).array().pow(1 / static_cast<Scalar>(dim));

    // The physical Hessians need the physical gradients, through the
    // curvature of the map
    if constexpr (dim == n) {
        if (flags_.physical_gradients || flags_.physical_hessians) {
            Map::physical_grad_on_element(tangents_, gradients_,
                                          physical_gradients_);
        }

        if (flags_.physical_hessians) {
            Map::physical_hess_on_element(tangents_, second_tangents_,
                                          physical_gradients_, hessians_,
                                          physical_hessians_);
        }
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::shift(const Eigen::MatrixX<Scalar>& shifts,
                                    int order)
    requires std::same_as<Basis, TensorBSpline<Scalar, dim>>
{
    basis_.taylor_on_element(first_active_, points_, shifts, order, values_);
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::differentiate(std::size_t direction)
    requires std::same_as<Basis, TensorBSpline<Scalar, dim>>
{
    const int degree = basis_.grid().knots(direction).degree();

    basis_.deriv_on_element(first_active_, points_, direction, degree,
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
