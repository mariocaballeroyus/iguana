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
 * @details Integrating the weak form by parts leaves the boundary term
 *          which pairs each trace with a flux, such as the field u of the
 *          Poisson problem with its normal derivative du/dn. Each element
 *          declares its fluxes next to its traces
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
     * @brief Flux of each degree of freedom of the element at the boundary
     *        points in it, the matrix F
     *
     *     F(u)(x_q) = Σ_i F_iq u_i
     *
     * @param values Values at the boundary points in the element
     * @param normals Unit normals of the boundary in physical space,
     *        pointing out of the domain, of size (num_points, n)
     * @param flux Output with a row per degree of freedom of the element,
     *        ordered as the rows of Element, and a column per point,
     *        overwritten. It is resized when necessary
     *
     * @pre @p values were filled with flags(), at the points of @p normals
     */
    virtual void local_flux(const ElementValues<Basis, n>& values,
                            const PointMatrix<Scalar, n>& normals,
                            Eigen::MatrixX<Scalar>& flux) const = 0;
};

} // namespace iguana

#endif // IGUANA_ELEMENT_FLUX_HPP
