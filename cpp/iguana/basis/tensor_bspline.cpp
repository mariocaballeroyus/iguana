/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "tensor_bspline.hpp"

#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
TensorBSpline<T, d>::TensorBSpline(std::array<BSpline<T>, d> axes)
    : axes_(std::move(axes)),
      num_functions_(1),
      num_elements_(1),
      num_active_(1)
{
    for (const BSpline<T>& axis : axes_) {
        num_functions_ *= axis.num_functions();
        num_elements_ *= axis.num_elements();
        num_active_ *= axis.num_active();
    }
}

template class TensorBSpline<double, 1>;
template class TensorBSpline<double, 2>;
template class TensorBSpline<double, 3>;

} // namespace iguana
