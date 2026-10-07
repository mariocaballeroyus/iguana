/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "assembler.hpp"

#include <concepts>
#include <span>
#include <stdexcept>
#include <vector>

#include "iguana/assembly/element_values.hpp"
#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/condition/neumann_condition.hpp"
#include "iguana/condition/penalty_condition.hpp"
#include "iguana/element/poisson/poisson_element.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Expands the values at the points of one element of a shifted
 *        quadrature towards the boundary it is shifted onto
 *
 * @throws std::invalid_argument If the patch is not a B-spline one, whose
 *         basis alone provides the Taylor series
 */
template<typename Basis, std::size_t n, std::size_t dim>
void shift_values(ElementValues<Basis, n>& values,
                  const BoundaryQuadrature<typename Basis::Scalar, dim>&
                      quadrature,
                  int first, int count)
{
    using Scalar = typename Basis::Scalar;

    if constexpr (std::same_as<Basis, TensorBSpline<Scalar, dim>>) {
        values.shift(quadrature.distances().middleRows(first, count),
                     quadrature.order());
    }
    else if constexpr (std::same_as<Basis, TensorNURBS<Scalar, dim>>) {
        throw std::invalid_argument("Assembler: "
                                    "a shifted quadrature needs a B-spline "
                                    "patch");
    }
}

} // namespace

template<typename Basis, std::size_t n>
Assembler<Basis, n>::Assembler(const FunctionSpace<Basis>& space,
                               const Patch<Basis, n>& patch)
    : space_(space), patch_(patch)
{
    const DofMap& dof_map = space.dof_map();
    const int num_dofs = dof_map.num_dofs();

    // Sparsity pattern: K_ij can only be non-zero when functions i and j
    // share an element, so each such pair gets an explicit zero, duplicates
    // merging into one entry
    std::vector<Eigen::Triplet<Scalar>> entries;

    for (int cell = 0; cell < dof_map.num_elements(); ++cell) {
        const std::span<const int> dofs = dof_map.dofs_on_element(cell);

        for (const int row : dofs) {
            for (const int col : dofs)
                entries.emplace_back(row, col, Scalar{0});
        }
    }

    stiffness_.resize(num_dofs, num_dofs);
    stiffness_.setFromTriplets(entries.begin(), entries.end());
    load_.setZero(num_dofs);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_stiffness(
    const E& element, const DomainQuadrature<Scalar, dim>& quadrature)
{
    ElementValues<Basis, n> values(patch_, element.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;
    Eigen::MatrixX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // Physical weights, the quadrature weights times the measure
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(values.measures());

        element.local_stiffness(values, weights, local);
        add_to_stiffness(cell, local);
    }
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_load(
    const E& element, const DomainQuadrature<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& source)
{
    if (source.size() != quadrature.num_points())
        throw std::invalid_argument("Assembler: "
                                    "the source must have one value per point");

    ElementValues<Basis, n> values(patch_, element.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;
    Eigen::VectorX<Scalar> sources;
    Eigen::VectorX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // Physical weights, the quadrature weights times the measure
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(values.measures());

        sources = source.segment(first, count);
        element.local_load(values, weights, sources, local);
        add_to_load(cell, local);
    }
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_stiffness(
    const C& condition, const BoundaryQuadrature<Scalar, dim>& quadrature)
{
    ElementValues<Basis, n> values(patch_, condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    Eigen::MatrixX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // A shifted quadrature expands the values towards the boundary it is
        // shifted onto, its weights staying those of its points
        if (quadrature.is_shifted())
            shift_values(values, quadrature, first, count);

        // Boundary weights, the quadrature weights times the measure of the
        // boundary through the map, from its normals in parameter space
        normals = quadrature.normals().middleRows(first, count);
        Patch<Basis, n>::boundary_measure_on_element(values.tangents(),
                                                     normals, measures);
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(measures);

        condition.local_stiffness(values, weights, local);
        add_to_stiffness(cell, local);
    }
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_load(
    const C& condition, const BoundaryQuadrature<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& data)
{
    if (data.size() != quadrature.num_points())
        throw std::invalid_argument("Assembler: "
                                    "the data must have one value per point");

    ElementValues<Basis, n> values(patch_, condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // A shifted quadrature expands the values towards the boundary it is
        // shifted onto, its weights staying those of its points
        if (quadrature.is_shifted())
            shift_values(values, quadrature, first, count);

        // Boundary weights, the quadrature weights times the measure of the
        // boundary through the map, from its normals in parameter space
        normals = quadrature.normals().middleRows(first, count);
        Patch<Basis, n>::boundary_measure_on_element(values.tangents(),
                                                     normals, measures);
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(measures);

        imposed = data.segment(first, count);
        condition.local_load(values, weights, imposed, local);
        add_to_load(cell, local);
    }
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::add_to_stiffness(int cell,
                                           const Eigen::MatrixX<Scalar>& local)
{
    // The k-th degree of freedom of the cell pairs with row k
    const std::span<const int> dofs = space_.dof_map().dofs_on_element(cell);

    for (std::size_t row = 0; row < dofs.size(); ++row) {
        for (std::size_t col = 0; col < dofs.size(); ++col)
            stiffness_.coeffRef(dofs[row], dofs[col]) += local(row, col);
    }
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::add_to_load(int cell,
                                      const Eigen::VectorX<Scalar>& local)
{
    // The k-th degree of freedom of the cell pairs with row k
    const std::span<const int> dofs = space_.dof_map().dofs_on_element(cell);

    for (std::size_t row = 0; row < dofs.size(); ++row)
        load_(dofs[row]) += local(row);
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::clear()
{
    // Only the stored values, so that the pattern stays
    stiffness_.coeffs().setZero();
    load_.setZero();
}

template class Assembler<TensorBSpline<double, 2>, 2>;
template class Assembler<TensorBSpline<double, 3>, 3>;
template class Assembler<TensorNURBS<double, 2>, 2>;
template class Assembler<TensorNURBS<double, 3>, 3>;

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const NeumannCondition<TensorBSpline<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const NeumannCondition<TensorNURBS<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const NeumannCondition<TensorBSpline<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const NeumannCondition<TensorNURBS<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&, const Eigen::VectorX<double>&);

} // namespace iguana
