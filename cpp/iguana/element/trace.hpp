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
 *        condition imposes values on or loads with natural data
 *
 * @details An essential quantity of the weak form, such as the field or a
 *          rotation, one per component of a vector quantity, whose rows of
 *          the other components are zero. Each element declares its traces
 *          as nested classes, such as PoissonElement::U. Projections onto
 *          the normal of a boundary belong to the conditions
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
     * @brief Field whose values the trace reads, an index into the bases of
     *        ElementValues
     *
     * It is the first, the only one of an element on one basis, unless the
     * element overrides it. A ghost penalty differentiates the values of
     * this field and takes the degree of its basis
     */
    virtual std::size_t field() const noexcept
    { return 0; }

    /**
     * @brief Trace of each degree of freedom of the element at the boundary
     *        points in it, the matrix B
     *
     *     T(u)(x_q) = Σ_i B_iq u_i
     *
     * @param values Values at the boundary points in the element
     * @param trace Output with a row per degree of freedom of the element,
     *        ordered as the rows of Element, and a column per point,
     *        overwritten. It is resized when necessary
     *
     * @pre @p values were filled with flags()
     */
    virtual void local_trace(const ElementValues<Basis, n>& values,
                             Eigen::MatrixX<Scalar>& trace) const = 0;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_TRACE_HPP
