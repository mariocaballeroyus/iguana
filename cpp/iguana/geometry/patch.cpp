/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "patch.hpp"

#include<stdexcept>
#include<utility>

namespace iguana
{

using Eigen::placeholders::all;

template<int d, std::floating_point T>
Patch<d, T>::Patch(TensorProductBSpline<d, T> basis,
                   Eigen::MatrixX3<T> coefficients)
    : basis_(std::move(basis)), coefficients_(std::move(coefficients))
{
    if (coefficients_.rows() != basis_.num_functions())
        throw std::invalid_argument("Patch: "
                                    "there must be one control point per "
                                    "basis function");
}

template<int d, std::floating_point T>
void Patch<d, T>::eval(const Eigen::VectorXi& actives,
                       const Eigen::MatrixX<T>& values,
                       Eigen::MatrixX3<T>& physical) const
{
    // Output buffer, reused across elements
    physical.resize(values.cols(), 3);

    // The active rows of the control points, gathered by the product
    physical.noalias() = values.transpose() * coefficients_(actives, all);
}

template class Patch<1, double>;
template class Patch<2, double>;
template class Patch<3, double>;

} // namespace iguana
