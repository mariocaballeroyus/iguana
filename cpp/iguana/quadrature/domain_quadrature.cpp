/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "domain_quadrature.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "iguana/domain/tensor_domain_iterator.hpp"
#include "iguana/quadrature/gauss_legendre.hpp"
#include "iguana/quadrature/quadrature_rule.hpp"

namespace iguana
{

template<std::floating_point T, std::size_t d>
DomainQuadrature<T, d>::DomainQuadrature()
    : offsets_(Eigen::VectorXi::Zero(1)),
      points_(0, d)
{
}

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

template<std::floating_point T, std::size_t d>
template<typename Rule>
void DomainQuadrature<T, d>::fill(const TensorDomain<T, d>& domain,
                                  CellType cell_type,
                                  const Rule& rule)
{
    // Elements already held, which the new cells must not repeat
    std::vector<bool> held(domain.num_elements(), false);

    for (const int element : elements_) {
        if (element < 0 || element >= domain.num_elements())
            throw std::invalid_argument("DomainQuadrature: "
                                        "the elements already held must lie "
                                        "in the domain");

        held[element] = true;
    }

    int num_cells = 0;

    for (int element = 0; element < domain.num_elements(); ++element) {
        if (domain.cell_type(element) != cell_type)
            continue;

        if (held[element])
            throw std::invalid_argument("DomainQuadrature: "
                                        "a cell of this type is already "
                                        "held");

        ++num_cells;
    }

    // The new cells follow the elements already held
    const int num_held = num_elements();

    Eigen::VectorXi elements(num_held + num_cells);
    Eigen::VectorXi offsets(num_held + num_cells + 1);
    elements.head(num_held) = elements_;
    offsets.head(num_held + 1) = offsets_;

    // Rule of each new cell, kept until the total number of points is known
    std::vector<Eigen::MatrixX<T>> cell_points(num_cells);
    std::vector<Eigen::VectorX<T>> cell_weights(num_cells);

    // Rule of the current cell on the reference cell, reused over the cells
    Eigen::MatrixX<T> reference_points;
    Eigen::VectorX<T> reference_weights;

    // Position of the current cell among the new ones
    int cell = 0;

    for (const TensorDomainIterator<T, d>& element : domain) {
        if (domain.cell_type(element.index()) != cell_type)
            continue;

        rule.reference_rule(element.start(), element.end(), reference_points,
                            reference_weights);
        map_to_cell(element.start(), element.end(), reference_points,
                    reference_weights, cell_points[cell], cell_weights[cell]);

        const int position = num_held + cell;

        elements(position) = element.index();
        offsets(position + 1) =
            offsets(position) + static_cast<int>(cell_weights[cell].size());
        ++cell;
    }

    const int num_total = offsets(num_held + num_cells);

    Eigen::MatrixX<T> points(num_total, d);
    Eigen::VectorX<T> weights(num_total);
    points.topRows(num_points()) = points_;
    weights.head(num_points()) = weights_;

    for (cell = 0; cell < num_cells; ++cell) {
        const int position = num_held + cell;
        const int count = offsets(position + 1) - offsets(position);

        points.middleRows(offsets(position), count) = cell_points[cell];
        weights.segment(offsets(position), count) = cell_weights[cell];
    }

    // Replace the arrays only once complete, so that a throwing rule leaves
    // the quadrature unchanged
    elements_ = std::move(elements);
    offsets_ = std::move(offsets);
    points_ = std::move(points);
    weights_ = std::move(weights);
}

template class DomainQuadrature<double, 1>;
template class DomainQuadrature<double, 2>;
template class DomainQuadrature<double, 3>;

template void DomainQuadrature<double, 1>::fill(
    const TensorDomain<double, 1>&, CellType, const GaussLegendre<double, 1>&);
template void DomainQuadrature<double, 2>::fill(
    const TensorDomain<double, 2>&, CellType, const GaussLegendre<double, 2>&);
template void DomainQuadrature<double, 3>::fill(
    const TensorDomain<double, 3>&, CellType, const GaussLegendre<double, 3>&);

} // namespace iguana
