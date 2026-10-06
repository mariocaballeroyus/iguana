/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_TRACE_HPP
#define IGUANA_ELEMENT_TRACE_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"

namespace iguana
{

/**
 * @brief Trace of the field of an element on a boundary, the quantity a
 *        condition imposes values on
 *
 * A trace is an essential quantity of the weak form of an element: one of
 * its degrees of freedom, or a combination of them such as a rotation.
 * Each element declares its traces as nested classes, such as
 * PoissonElement::U for the field of the Poisson problem. Fluxes are
 * natural, so they have no trace. A trace gives one value per
 * boundary point, linear in the coefficients of the element,
 * T(u)(x_q) = sum over i of B_iq u_i, so that a vector quantity is imposed
 * through one trace per component
 *
 * A trace belongs to the element alone and needs no boundary. Projections
 * onto the normal or the tangent of the boundary, such as a normal
 * displacement, belong to the conditions, which combine the traces of the
 * components with the normals
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class Trace
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    virtual ~Trace() = default;

    /// @brief Optional values the trace reads from ElementValues
    virtual ValueFlags flags() const noexcept = 0;

    /**
     * @brief Trace of each active function at the boundary points in the
     *        element, the matrix B
     *
     * @param values Values at the boundary points in the element
     * @param trace Output of size (num_active, num_points), the layout of
     *        ElementValues::values(), overwritten. It is resized when
     *        necessary
     *
     * @pre @p values were filled with flags()
     */
    virtual void local_trace(const ElementValues<Basis, n>& values,
                             Eigen::MatrixX<Scalar>& trace) const = 0;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_TRACE_HPP
