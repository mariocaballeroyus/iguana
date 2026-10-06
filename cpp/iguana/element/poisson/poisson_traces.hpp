/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_POISSON_POISSON_TRACES_HPP
#define IGUANA_ELEMENT_POISSON_POISSON_TRACES_HPP

#include <cstddef>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/element/poisson/poisson_element.hpp"
#include "iguana/element/trace.hpp"

namespace iguana
{

/**
 * @brief Trace of the field u of the Poisson problem, its only essential
 *        quantity, the flux being natural
 *
 * Its matrix B holds the values of the active functions at the boundary
 * points
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space, the number of parametric
 *         directions
 */
template<typename Basis, std::size_t n>
class PoissonElement<Basis, n>::U final : public Trace<Basis, n>
{
public:
    ValueFlags flags() const noexcept override
    { return {}; }

    void local_trace(const ElementValues<Basis, n>& values,
                     Eigen::MatrixX<Scalar>& trace) const override
    { trace = values.values(); }
};

} // namespace iguana

#endif // IGUANA_ELEMENT_POISSON_POISSON_TRACES_HPP
