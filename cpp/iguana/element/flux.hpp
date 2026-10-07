/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_FLUX_HPP
#define IGUANA_ELEMENT_FLUX_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Flux of the field of an element through a boundary, the natural
 *        quantity conjugate to one of its traces
 *
 * Integrating the weak form of an element by parts leaves a boundary term
 * that pairs each trace T with a flux F, the integral of F(u) T(v) ds, such
 * as the normal derivative du/dn with the field u of the Poisson problem.
 * Conditions that keep this term, such as Nitsche's, read the flux, while a
 * Neumann condition replaces it by data. Each element declares its fluxes
 * as nested classes next to its traces, such as PoissonElement::Flux. A
 * flux gives one value per boundary point, linear in the coefficients of
 * the element, F(u)(x_q) = sum over i of F_iq u_i. Unlike a trace, it
 * needs the normals of the boundary
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class Flux
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    virtual ~Flux() = default;

    /// @brief Optional values the flux reads from ElementValues
    virtual ValueFlags flags() const noexcept = 0;

    /**
     * @brief Flux of each active function at the boundary points in the
     *        element, the matrix F
     *
     * @param values Values at the boundary points in the element
     * @param normals Unit normals of the boundary in physical space,
     *        pointing out of the domain, of size (num_points, n)
     * @param flux Output of size (num_active, num_points), the layout of
     *        ElementValues::values(), overwritten. It is resized when
     *        necessary
     *
     * @pre @p values were filled with flags(), at the points of @p normals
     */
    virtual void local_flux(const ElementValues<Basis, n>& values,
                            const PointMatrix<Scalar, n>& normals,
                            Eigen::MatrixX<Scalar>& flux) const = 0;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_FLUX_HPP
