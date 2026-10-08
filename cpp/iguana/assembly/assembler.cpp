/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "assembler.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "iguana/assembly/element_values.hpp"
#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/condition/neumann_condition.hpp"
#include "iguana/condition/nitsche_condition.hpp"
#include "iguana/condition/penalty_condition.hpp"
#include "iguana/element/poisson/poisson_element.hpp"
#include "iguana/utils/multi_index.hpp"

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

/**
 * @brief Differentiates the values at the points of one face across it, to
 *        the order of the degree there
 *
 * @throws std::invalid_argument If the patch is not a B-spline one, whose
 *         basis alone provides the derivatives
 */
template<typename Basis, std::size_t n>
void differentiate_values(ElementValues<Basis, n>& values,
                          std::size_t direction)
{
    using Scalar = typename Basis::Scalar;
    constexpr std::size_t dim = Basis::dimension;

    if constexpr (std::same_as<Basis, TensorBSpline<Scalar, dim>>) {
        values.differentiate(direction);
    }
    else if constexpr (std::same_as<Basis, TensorNURBS<Scalar, dim>>) {
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty needs a B-spline patch");
    }
}

} // namespace

template<typename Basis, std::size_t n>
Assembler<Basis, n>::Assembler(const FunctionSpace<Basis>& space,
                               const Patch<Basis, n>& patch,
                               bool couple_faces)
    : space_(space), patch_(patch), couple_faces_(couple_faces)
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

    // A term on a face pairs the functions of its two cells, so with the
    // faces coupled, every interior face of the grid adds those pairs, each
    // face found once from the cell before it
    if (couple_faces) {
        std::array<int, dim> counts{};

        for (std::size_t direction = 0; direction < dim; ++direction)
            counts[direction] =
                space.basis().grid().knots(direction).num_elements();

        for (int cell = 0; cell < dof_map.num_elements(); ++cell) {
            const std::array<int, dim> index = unflatten(cell, counts);
            const std::span<const int> before = dof_map.dofs_on_element(cell);

            for (std::size_t direction = 0; direction < dim; ++direction) {
                std::array<int, dim> next = index;
                next[direction] += 1;

                if (next[direction] == counts[direction])
                    continue;

                const std::span<const int> after =
                    dof_map.dofs_on_element(flatten(next, counts));

                for (const int row : before) {
                    for (const int col : after) {
                        entries.emplace_back(row, col, Scalar{0});
                        entries.emplace_back(col, row, Scalar{0});
                    }
                }
            }
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
    // Every cell weighted by one
    const Eigen::VectorX<Scalar> ones =
        Eigen::VectorX<Scalar>::Ones(space_.dof_map().num_elements());

    assemble_stiffness(element, quadrature, ones);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_stiffness(
    const E& element, const DomainQuadrature<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& cell_weights)
{
    if (cell_weights.size() != space_.dof_map().num_elements())
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");

    ElementValues<Basis, n> values(patch_, element.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;
    Eigen::MatrixX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const Scalar cell_weight = cell_weights(cell);

        // A cell of weight zero adds nothing
        if (cell_weight == 0)
            continue;

        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // Physical weights, the quadrature weights times the measure, times
        // the weight of the test functions of the cell, constant on it
        weights = quadrature.weights().segment(first, count);
        weights = cell_weight * weights.cwiseProduct(values.measures());

        element.local_stiffness(values, weights, local);
        add_to_stiffness(space_.dof_map().dofs_on_element(cell), local);
    }
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_load(
    const E& element, const DomainQuadrature<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& source)
{
    // Every cell weighted by one
    const Eigen::VectorX<Scalar> ones =
        Eigen::VectorX<Scalar>::Ones(space_.dof_map().num_elements());

    assemble_load(element, quadrature, source, ones);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_load(
    const E& element, const DomainQuadrature<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& source,
    const Eigen::VectorX<Scalar>& cell_weights)
{
    if (source.size() != quadrature.num_points())
        throw std::invalid_argument("Assembler: "
                                    "the source must have one value per point");

    if (cell_weights.size() != space_.dof_map().num_elements())
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");

    ElementValues<Basis, n> values(patch_, element.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;
    Eigen::VectorX<Scalar> sources;
    Eigen::VectorX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const Scalar cell_weight = cell_weights(cell);

        // A cell of weight zero adds nothing
        if (cell_weight == 0)
            continue;

        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // Physical weights, the quadrature weights times the measure, times
        // the weight of the test functions of the cell, constant on it
        weights = quadrature.weights().segment(first, count);
        weights = cell_weight * weights.cwiseProduct(values.measures());

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
    ElementValues<Basis, n> shifted_values(patch_, condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;
    Eigen::MatrixX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // A shifted quadrature also expands the values towards the boundary
        // it is shifted onto, its weights and normals staying those of its
        // points. Otherwise the shifted values are the values themselves
        if (quadrature.is_shifted()) {
            shifted_values.reinit(cell, points);
            shift_values(shifted_values, quadrature, first, count);
        }

        const ElementValues<Basis, n>& shifted =
            quadrature.is_shifted() ? shifted_values : values;

        // Boundary weights, the quadrature weights times the measure of the
        // boundary through the map, and the normals in physical space, from
        // the normals in parameter space
        normals = quadrature.normals().middleRows(first, count);
        Patch<Basis, n>::boundary_measure_on_element(values.tangents(),
                                                     normals, measures);
        Patch<Basis, n>::physical_normal_on_element(values.tangents(),
                                                    normals,
                                                    physical_normals);
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(measures);

        condition.local_stiffness(values, shifted, weights, physical_normals,
                                  local);
        add_to_stiffness(space_.dof_map().dofs_on_element(cell), local);
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
    ElementValues<Basis, n> shifted_values(patch_, condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> local;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // A shifted quadrature also expands the values towards the boundary
        // it is shifted onto, its weights and normals staying those of its
        // points. Otherwise the shifted values are the values themselves
        if (quadrature.is_shifted()) {
            shifted_values.reinit(cell, points);
            shift_values(shifted_values, quadrature, first, count);
        }

        const ElementValues<Basis, n>& shifted =
            quadrature.is_shifted() ? shifted_values : values;

        // Boundary weights, the quadrature weights times the measure of the
        // boundary through the map, and the normals in physical space, from
        // the normals in parameter space
        normals = quadrature.normals().middleRows(first, count);
        Patch<Basis, n>::boundary_measure_on_element(values.tangents(),
                                                     normals, measures);
        Patch<Basis, n>::physical_normal_on_element(values.tangents(),
                                                    normals,
                                                    physical_normals);
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(measures);

        imposed = data.segment(first, count);
        condition.local_load(values, shifted, weights, physical_normals,
                             imposed, local);
        add_to_load(cell, local);
    }
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::assemble_stiffness(
    const GhostPenalty<Basis, n>& penalty,
    const FaceQuadrature<Scalar, dim>& quadrature) requires (dim == 2)
{
    // Without the faces coupled, the pattern would grow with each new face
    if (!couple_faces_)
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty needs the faces coupled");

    // Only on an affine map do the jumps of the derivatives across the knot
    // lines give those of the physical normal derivatives
    if (!patch_.is_affine())
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty needs an affine patch");

    const DofMap& dof_map = space_.dof_map();
    ElementValues<Basis, n> before(patch_, penalty.flags());
    ElementValues<Basis, n> after(patch_, penalty.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

    for (int face = 0; face < quadrature.num_faces(); ++face) {
        const int first = quadrature.offsets()(face);
        const int count = quadrature.offsets()(face + 1) - first;
        const int cell_before = quadrature.cells()(face, 0);
        const int cell_after = quadrature.cells()(face, 1);
        const std::size_t direction =
            static_cast<std::size_t>(quadrature.directions()(face));
        const int degree = patch_.basis().grid().knots(direction).degree();

        // The values of each cell, with its own polynomial, differentiated
        // across the face at the same points
        points = quadrature.points().middleRows(first, count);
        before.reinit(cell_before, points);
        after.reinit(cell_after, points);
        differentiate_values(before, direction);
        differentiate_values(after, direction);

        // Face weights, the quadrature weights times the measure of the face
        // through the map, which is continuous across it
        normals = quadrature.normals().middleRows(first, count);
        Patch<Basis, n>::boundary_measure_on_element(before.tangents(),
                                                     normals, measures);
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(measures);

        penalty.local_stiffness(before, after, weights, degree, local);

        // The rows follow the cell before the face, then the cell after it
        const std::span<const int> before_dofs =
            dof_map.dofs_on_element(cell_before);
        const std::span<const int> after_dofs =
            dof_map.dofs_on_element(cell_after);
        dofs.assign(before_dofs.begin(), before_dofs.end());
        dofs.insert(dofs.end(), after_dofs.begin(), after_dofs.end());

        add_to_stiffness(dofs, local);
    }
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::add_to_stiffness(std::span<const int> dofs,
                                           const Eigen::MatrixX<Scalar>& local)
{
    // The k-th degree of freedom pairs with row k
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
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const DomainQuadrature<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const DomainQuadrature<double, 3>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&);

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

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const BoundaryQuadrature<double, 2>&, const Eigen::VectorX<double>&);

} // namespace iguana
