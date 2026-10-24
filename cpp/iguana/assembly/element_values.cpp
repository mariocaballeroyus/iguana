/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "element_values.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

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
    : ElementValues(patch, std::vector{std::cref(basis)}, flags)
{
}

template<typename Basis, std::size_t n>
ElementValues<Basis, n>::ElementValues(
    const Patch<Basis, n>& patch,
    const std::vector<std::reference_wrapper<const Basis>>& bases,
    ValueFlags flags)
    : patch_(patch), flags_(flags), map_(patch.basis())
{
    if (bases.empty())
        throw std::invalid_argument("ElementValues: "
                                    "the values need at least one basis");

    // One element index locates the points in every basis, which share
    // their knot lines with the patch
    const std::array<std::vector<Scalar>, dim> lines =
        patch.basis().grid().lines();

    fields_.reserve(bases.size());

    for (const Basis& basis : bases) {
        if (basis.grid().lines() != lines)
            throw std::invalid_argument("ElementValues: "
                                        "every basis must have the elements "
                                        "of the patch");

        if (!map_field_ && basis == patch.basis())
            map_field_ = fields_.size();

        fields_.emplace_back(basis);
    }

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

    // Volume of the element in parameter space, the same in every basis
    int remaining = element;
    Scalar volume = 1;

    for (std::size_t direction = 0; direction < dim; ++direction) {
        const KnotVector<Scalar>& knots =
            patch_.basis().grid().knots(direction);
        const int axis_element = remaining % knots.num_elements();

        volume *= knots.element_end(axis_element)
                  - knots.element_start(axis_element);
        remaining /= knots.num_elements();
    }

    // The points stay for a later shift
    points_ = points;

    for (Field& field : fields_) {
        evaluate(field, element, points);
        field.combinations.clear();
    }

    // The patch's own functions give the map, which are those of a field
    // when its basis is the patch's
    if (!map_field_)
        evaluate(map_, element, points);

    const Field& map = map_field_ ? fields_[*map_field_] : map_;
    patch_.tangent_on_element(map.actives, map.gradients, tangents_);

    if (flags_.physical_hessians)
        patch_.second_tangent_on_element(map.actives, map.hessians,
                                         second_tangents_);

    Map::measure_on_element(tangents_, measures_);
    sizes_ = (volume * measures_).array().pow(1 / static_cast<Scalar>(dim));

    // The physical Hessians need the physical gradients, through the
    // curvature of the map
    if constexpr (dim == n) {
        for (Field& field : fields_) {
            if (flags_.physical_gradients || flags_.physical_hessians) {
                Map::physical_grad_on_element(tangents_, field.gradients,
                                              field.physical_gradients);
            }

            if (flags_.physical_hessians) {
                Map::physical_hess_on_element(tangents_, second_tangents_,
                                              field.physical_gradients,
                                              field.hessians,
                                              field.physical_hessians);
            }
        }
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::shift(const Eigen::MatrixX<Scalar>& shifts,
                                    int order)
    requires std::same_as<Basis, TensorBSpline<Scalar, dim>>
{
    for (Field& field : fields_) {
        field.basis.taylor_on_element(field.first_active, points_, shifts,
                                      order, field.values);
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::shift(
    const Eigen::MatrixX<Scalar>& shifts,
    const Eigen::MatrixX<Scalar>& weights,
    const std::vector<Eigen::MatrixX<Scalar>>& coupled, int order,
    ShiftCenter center)
    requires std::same_as<Basis, TensorBSpline<Scalar, dim>>
{
    // From the surrogate, the combinations add to the Taylor series of the
    // values and take the derivatives at the points themselves
    if (center == ShiftCenter::surrogate)
        shift(shifts, order);

    const Eigen::MatrixX<Scalar> at =
        center == ShiftCenter::surrogate
            ? points_
            : Eigen::MatrixX<Scalar>(points_ + shifts);
    Eigen::MatrixX<Scalar> combination;

    for (Field& field : fields_) {
        field.basis.combine_on_element(field.first_active, at, weights, order,
                                       combination);
        field.values += combination;

        // The further combinations, apart from the values
        field.combinations.resize(coupled.size());

        for (std::size_t set = 0; set < coupled.size(); ++set) {
            field.basis.combine_on_element(field.first_active, at,
                                           coupled[set], order,
                                           field.combinations[set]);
        }
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::differentiate(std::size_t direction)
    requires std::same_as<Basis, TensorBSpline<Scalar, dim>>
{
    // |J^-T e_k| is the measure of a knot line of the direction over that of
    // the patch, by Nanson's formula with the normal e_k
    Eigen::MatrixX<Scalar> normals =
        Eigen::MatrixX<Scalar>::Zero(points_.rows(), dim);
    normals.col(static_cast<Eigen::Index>(direction)).setOnes();

    Eigen::VectorX<Scalar> line_measures;
    Patch<Basis, n>::boundary_measure_on_element(tangents_, normals,
                                                 line_measures);

    for (Field& field : fields_) {
        const int degree = field.basis.grid().knots(direction).degree();

        field.basis.deriv_on_element(field.first_active, points_, direction,
                                     degree, field.values);

        const Eigen::VectorX<Scalar> scales =
            (line_measures.array() / measures_.array())
                .pow(static_cast<Scalar>(degree));
        field.values *= scales.asDiagonal();
    }
}

template<typename Basis, std::size_t n>
void ElementValues<Basis, n>::evaluate(Field& field, int element,
                                       const Eigen::MatrixX<Scalar>& points)
{
    field.first_active = first_active_on(field.basis, element);
    field.basis.active_on_element(element, field.actives);

    // One evaluation gives the second derivatives with the first
    if (flags_.physical_hessians) {
        field.basis.hess_on_element(field.first_active, points, field.values,
                                    field.gradients, field.hessians);
    } else {
        field.basis.grad_on_element(field.first_active, points, field.values,
                                    field.gradients);
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
