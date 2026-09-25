/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "domain_quadrature.hpp"

#include <stdexcept>
#include <utility>

namespace iguana
{

template<std::floating_point T, std::size_t d>
DomainQuadrature<T, d>::DomainQuadrature(Eigen::VectorXi elements,
                                         Eigen::VectorXi offsets,
                                         Eigen::MatrixX<T> points,
                                         Eigen::VectorX<T> weights)
    : elements_(std::move(elements)),
      offsets_(std::move(offsets)),
      points_(std::move(points)),
      weights_(std::move(weights))
{
    if (points_.cols() != static_cast<Eigen::Index>(d))
        throw std::invalid_argument("DomainQuadrature: "
                                    "the points must have d columns");

    if (weights_.size() != points_.rows())
        throw std::invalid_argument("DomainQuadrature: "
                                    "there must be one weight per point");

    if (offsets_.size() != elements_.size() + 1)
        throw std::invalid_argument("DomainQuadrature: "
                                    "there must be one offset more than "
                                    "elements");

    // Element k owns the points from offsets(k) up to offsets(k + 1)
    const int count = num_elements();
    const bool ordered =
        offsets_(0) == 0 && offsets_(count) == points_.rows() &&
        (offsets_.tail(count).array() >= offsets_.head(count).array()).all();

    if (!ordered)
        throw std::invalid_argument("DomainQuadrature: "
                                    "the offsets must run from zero to the "
                                    "number of points without decreasing");
}

template class DomainQuadrature<double, 1>;
template class DomainQuadrature<double, 2>;
template class DomainQuadrature<double, 3>;

} // namespace iguana
