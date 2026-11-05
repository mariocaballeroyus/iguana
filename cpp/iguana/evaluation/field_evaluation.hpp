/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EVALUATION_FIELD_EVALUATION_HPP
#define IGUANA_EVALUATION_FIELD_EVALUATION_HPP

#include <Eigen/Core>

#include "iguana/element/flux.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/fspace/mixed_function_space.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Values at points of a trace of a field, such as a component of the
 *        velocity or the pressure of a flow
 *
 *     T(u)(x) = Σ_i B_i(x) u_i
 *
 * @details Each point takes the element holding it, which on a knot line is
 *          the one after the line. Only affine maps are inverted so far
 *
 * @tparam Basis Basis of the patch and of the fields, TensorBSpline or
 *         TensorNURBS, with as many directions as the physical space
 *
 * @param space Space of the field, a plain space being a mixed one of one
 *        field
 * @param patch Patch that maps the points
 * @param trace Trace to evaluate, of the element whose unknowns the space
 *        holds
 * @param coefficients Coefficients of the field, one per degree of freedom
 *        of the space
 * @param points Points in physical space, one per row
 * @return Value of the trace at each point
 *
 * @throws std::invalid_argument If the coefficients are not one per degree
 *         of freedom, if the map is not affine, if the space does not have
 *         the elements of the patch, if a point lies outside the patch or on
 *         a cell outside the physical domain, or if the trace has other rows
 *         than the degrees of freedom of an element
 */
template<typename Basis>
Eigen::VectorX<typename Basis::Scalar> trace_at_points(
    const MixedFunctionSpace<Basis>& space,
    const Patch<Basis, Basis::dimension>& patch,
    const Trace<Basis, Basis::dimension>& trace,
    const Eigen::VectorX<typename Basis::Scalar>& coefficients,
    const PointMatrix<typename Basis::Scalar, Basis::dimension>& points);

/**
 * @brief Values at points of a flux of a field, such as a component of the
 *        traction of a flow
 *
 *     F(u)(x) = Σ_i F_i(x) u_i
 *
 * @details Each point takes the element holding it, so that a flux jumping
 *          across a knot line, as at degree one, gives the value of the
 *          element after the line. The normals need not be those of a
 *          boundary of the field: the traction of a fluid along the normal
 *          of a solid, pointing into the fluid, is the load on the solid
 *
 * @tparam Basis Basis of the patch and of the fields, TensorBSpline or
 *         TensorNURBS, with as many directions as the physical space
 *
 * @param space Space of the field, a plain space being a mixed one of one
 *        field
 * @param patch Patch that maps the points
 * @param flux Flux to evaluate, of the element whose unknowns the space
 *        holds
 * @param coefficients Coefficients of the field, one per degree of freedom
 *        of the space
 * @param points Points in physical space, one per row
 * @param normals Unit normals in physical space, one per point
 * @return Value of the flux at each point
 *
 * @throws std::invalid_argument If the coefficients are not one per degree
 *         of freedom, if the normals are not one per point, if the map is
 *         not affine, if the space does not have the elements of the patch,
 *         if a point lies outside the patch or on a cell outside the
 *         physical domain, or if the flux has other rows than the degrees of
 *         freedom of an element
 */
template<typename Basis>
Eigen::VectorX<typename Basis::Scalar> flux_at_points(
    const MixedFunctionSpace<Basis>& space,
    const Patch<Basis, Basis::dimension>& patch,
    const Flux<Basis, Basis::dimension>& flux,
    const Eigen::VectorX<typename Basis::Scalar>& coefficients,
    const PointMatrix<typename Basis::Scalar, Basis::dimension>& points,
    const PointMatrix<typename Basis::Scalar, Basis::dimension>& normals);

} // namespace iguana

#endif // IGUANA_EVALUATION_FIELD_EVALUATION_HPP
