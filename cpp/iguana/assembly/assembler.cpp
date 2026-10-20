/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "assembler.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/SparseCore>

#include "iguana/assembly/element_values.hpp"
#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/condition/neumann_condition.hpp"
#include "iguana/condition/nitsche_condition.hpp"
#include "iguana/condition/penalty_condition.hpp"
#include "iguana/condition/skeleton_nitsche_condition.hpp"
#include "iguana/element/poisson/poisson_element.hpp"
#include "iguana/element/stokes/stokes_element.hpp"
#include "iguana/element/stokes/stokes_stabilized_element.hpp"
#include "iguana/grid/knot_vector.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Checks the error code of a PETSc call, as this build of PETSc has
 *        no C++ error handler of its own
 *
 * @throws std::runtime_error With the message of PETSc if the call failed
 */
void check_petsc(PetscErrorCode code)
{
    if (code == PETSC_SUCCESS)
        return;

    const char* message = nullptr;
    static_cast<void>(PetscErrorMessage(code, &message, nullptr));

    throw std::runtime_error(std::string("Assembler: ") + message);
}

/**
 * @brief Expands the values at the points of one element or face towards
 *        the boundary they are projected onto, along their distance vectors
 *
 * @throws std::invalid_argument If the patch is not a B-spline one, whose
 *         basis alone provides the Taylor series
 */
template<typename Basis, std::size_t n>
void shift_values(ElementValues<Basis, n>& values,
                  const Eigen::MatrixX<typename Basis::Scalar>& distances,
                  int order)
{
    using Scalar = typename Basis::Scalar;
    constexpr std::size_t dim = Basis::dimension;

    if constexpr (std::same_as<Basis, TensorBSpline<Scalar, dim>>) {
        values.shift(distances, order);
    }
    else if constexpr (std::same_as<Basis, TensorNURBS<Scalar, dim>>) {
        throw std::invalid_argument("Assembler: "
                                    "a projection needs a B-spline patch");
    }
}

/**
 * @brief Checks that a projection and an order can shift the values at the
 *        points of a quadrature
 *
 * @throws std::invalid_argument If the order is negative, or if the
 *         projection does not have one distance vector per point
 */
template<std::floating_point T>
void check_projection(const BoundaryProjection<T>& projection, int order,
                      int num_points)
{
    if (order < 0)
        throw std::invalid_argument("Assembler: "
                                    "the order must be non-negative");

    if (projection.distances.rows() != num_points)
        throw std::invalid_argument("Assembler: "
                                    "the projection must have one distance "
                                    "vector per point");
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

/**
 * @brief Whether every interior knot of a knot vector is simple, so that
 *        its splines have maximal continuity and only their derivatives of
 *        the order of the degree jump between its elements
 */
template<std::floating_point T>
bool simple_interior_knots(const KnotVector<T>& knot_vector)
{
    const std::vector<T>& knots = knot_vector.values();

    for (int element = 1; element < knot_vector.num_elements(); ++element) {
        const T knot = knot_vector.element_start(element);

        if (std::count(knots.begin(), knots.end(), knot) > 1)
            return false;
    }

    return true;
}

} // namespace

template<typename Basis, std::size_t n>
Assembler<Basis, n>::Assembler(const FunctionSpace<Basis>& space,
                               const Patch<Basis, n>& patch,
                               bool couple_faces)
    : Assembler(MixedFunctionSpace<Basis>(std::vector{std::cref(space)}),
                patch, couple_faces)
{
}

template<typename Basis, std::size_t n>
Assembler<Basis, n>::Assembler(const MixedFunctionSpace<Basis>& space,
                               const Patch<Basis, n>& patch,
                               bool couple_faces)
    : space_(space), patch_(patch), couple_faces_(couple_faces)
{
    // The values locate the points of each element in every basis, checked
    // before the system exists, as nothing would destroy it. The fields
    // share the elements of the first
    if (space.field(0).basis().grid().lines() != patch.basis().grid().lines())
        throw std::invalid_argument("Assembler: "
                                    "the space must have the elements of the "
                                    "patch");

    const int num_elements = space.num_elements();
    const int num_dofs = space.num_dofs();

    // Sparsity pattern: K_ij can only be non-zero when functions i and j
    // share an element, so each pair of their components gets an explicit
    // zero, duplicates merging into one entry
    std::vector<Eigen::Triplet<Scalar>> entries;
    std::vector<int> dofs;

    for (int cell = 0; cell < num_elements; ++cell) {
        space.dofs_on_element(cell, dofs);

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
                patch.basis().grid().knots(direction).num_elements();

        std::vector<int> after;

        for (int cell = 0; cell < num_elements; ++cell) {
            const std::array<int, dim> index = unflatten(cell, counts);
            space.dofs_on_element(cell, dofs);

            for (std::size_t direction = 0; direction < dim; ++direction) {
                std::array<int, dim> next = index;
                next[direction] += 1;

                if (next[direction] == counts[direction])
                    continue;

                space.dofs_on_element(flatten(next, counts), after);

                for (const int row : dofs) {
                    for (const int col : after) {
                        entries.emplace_back(row, col, Scalar{0});
                        entries.emplace_back(col, row, Scalar{0});
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> pattern(num_dofs, num_dofs);
    pattern.setFromTriplets(entries.begin(), entries.end());

    // The compressed rows of the pattern preallocate the stiffness, its
    // explicit zeros the only entries a write may reach. The block size
    // tells the solvers which degrees of freedom are components of one
    // function, in blocks only when the space has a single field
    const int block_size =
        space.num_fields() == 1 ? space.field(0).block_size() : 1;

    check_petsc(MatCreate(PETSC_COMM_SELF, &stiffness_));
    check_petsc(MatSetSizes(stiffness_, num_dofs, num_dofs, num_dofs,
                            num_dofs));
    check_petsc(MatSetBlockSize(stiffness_, block_size));
    check_petsc(MatSetType(stiffness_, MATSEQAIJ));
    check_petsc(MatSeqAIJSetPreallocationCSR(
        stiffness_, pattern.outerIndexPtr(), pattern.innerIndexPtr(),
        nullptr));

    // The mass shares the pattern, and both read local matrices by columns,
    // as Eigen stores them, and refuse entries outside the pattern
    check_petsc(MatDuplicate(stiffness_, MAT_SHARE_NONZERO_PATTERN, &mass_));

    for (Mat matrix : {stiffness_, mass_}) {
        check_petsc(MatSetOption(matrix, MAT_ROW_ORIENTED, PETSC_FALSE));
        check_petsc(MatSetOption(matrix, MAT_NEW_NONZERO_ALLOCATION_ERR,
                                 PETSC_TRUE));
    }

    // The load follows the rows, with their block size
    check_petsc(MatCreateVecs(stiffness_, nullptr, &load_));
    check_petsc(VecZeroEntries(load_));
}

template<typename Basis, std::size_t n>
Assembler<Basis, n>::~Assembler()
{
    // A destructor must not throw, so a failure is left to PETSc to report
    static_cast<void>(MatDestroy(&mass_));
    static_cast<void>(MatDestroy(&stiffness_));
    static_cast<void>(VecDestroy(&load_));
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_stiffness(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature)
{
    // Every cell weighted by one
    const Eigen::VectorX<Scalar> ones =
        Eigen::VectorX<Scalar>::Ones(space_.num_elements());

    assemble_stiffness(element, quadrature, ones);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_stiffness(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& cell_weights)
{
    if (cell_weights.size() != space_.num_elements())
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");

    ElementValues<Basis, n> values(patch_, space_.bases(), element.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;
    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

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
        space_.dofs_on_element(cell, dofs);
        add_to_matrix(stiffness_, dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_load(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::MatrixX<Scalar>& source)
{
    // Every cell weighted by one
    const Eigen::VectorX<Scalar> ones =
        Eigen::VectorX<Scalar>::Ones(space_.num_elements());

    assemble_load(element, quadrature, source, ones);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_load(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& cell_weights)
{
    if (source.rows() != quadrature.num_points())
        throw std::invalid_argument("Assembler: "
                                    "the source must have one row per point");

    if (cell_weights.size() != space_.num_elements())
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");

    ElementValues<Basis, n> values(patch_, space_.bases(), element.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;
    Eigen::MatrixX<Scalar> sources;
    Eigen::VectorX<Scalar> local;
    std::vector<int> dofs;

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

        sources = source.middleRows(first, count);
        element.local_load(values, weights, sources, local);
        space_.dofs_on_element(cell, dofs);
        add_to_load(dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_mass(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature)
{
    // Every cell weighted by one
    const Eigen::VectorX<Scalar> ones =
        Eigen::VectorX<Scalar>::Ones(space_.num_elements());

    assemble_mass(element, quadrature, ones);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_mass(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& cell_weights)
{
    if (cell_weights.size() != space_.num_elements())
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");

    ElementValues<Basis, n> values(patch_, space_.bases(), element.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;
    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

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

        element.local_mass(values, weights, local);
        space_.dofs_on_element(cell, dofs);
        add_to_matrix(mass_, dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_stiffness(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature)
{
    add_condition_stiffness(condition, quadrature, nullptr, 0);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_stiffness(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order)
    requires (dim == 2)
{
    add_condition_stiffness(condition, quadrature, &projection, order);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::add_condition_stiffness(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order)
{
    if (projection != nullptr)
        check_projection(*projection, order, quadrature.num_points());

    ElementValues<Basis, n> values(patch_, space_.bases(), condition.flags());
    ElementValues<Basis, n> shifted_values(patch_, space_.bases(),
                                           condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;
    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // With a projection, the values are also expanded towards the
        // boundary along the distance vectors, the weights and normals
        // staying those of the points. Otherwise the shifted values are the
        // values themselves
        if (projection != nullptr) {
            shifted_values.reinit(cell, points);
            shift_values(shifted_values,
                         projection->distances.middleRows(first, count),
                         order);
        }

        const ElementValues<Basis, n>& shifted =
            projection != nullptr ? shifted_values : values;

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
        space_.dofs_on_element(cell, dofs);
        add_to_matrix(stiffness_, dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_load(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& data)
{
    add_condition_load(condition, quadrature, nullptr, 0, data);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_load(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order,
    const Eigen::VectorX<Scalar>& data) requires (dim == 2)
{
    add_condition_load(condition, quadrature, &projection, order, data);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::add_condition_load(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order,
    const Eigen::VectorX<Scalar>& data)
{
    if (data.size() != quadrature.num_points())
        throw std::invalid_argument("Assembler: "
                                    "the data must have one value per point");

    if (projection != nullptr)
        check_projection(*projection, order, quadrature.num_points());

    ElementValues<Basis, n> values(patch_, space_.bases(), condition.flags());
    ElementValues<Basis, n> shifted_values(patch_, space_.bases(),
                                           condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> local;
    std::vector<int> dofs;

    for (int held = 0; held < quadrature.num_elements(); ++held) {
        const int cell = quadrature.elements()(held);
        const int first = quadrature.offsets()(held);
        const int count = quadrature.offsets()(held + 1) - first;

        points = quadrature.points().middleRows(first, count);
        values.reinit(cell, points);

        // With a projection, the values are also expanded towards the
        // boundary along the distance vectors, the weights and normals
        // staying those of the points. Otherwise the shifted values are the
        // values themselves
        if (projection != nullptr) {
            shifted_values.reinit(cell, points);
            shift_values(shifted_values,
                         projection->distances.middleRows(first, count),
                         order);
        }

        const ElementValues<Basis, n>& shifted =
            projection != nullptr ? shifted_values : values;

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
        space_.dofs_on_element(cell, dofs);
        add_to_load(dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::assemble_stiffness(
    const SkeletonGhostPenalty<Basis, n>& penalty,
    const QuadratureSkeleton<Scalar, dim>& quadrature) requires (dim == 2)
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

    // The penalty extends one field, whose basis gives the degrees
    if (penalty.field() >= space_.num_fields())
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty must act on a field of "
                                    "the space");

    const Basis& basis = space_.field(penalty.field()).basis();

    // A repeated knot of the field would let lower derivatives of its
    // functions jump unpenalized
    for (std::size_t direction = 0; direction < dim; ++direction) {
        if (!simple_interior_knots(basis.grid().knots(direction)))
            throw std::invalid_argument("Assembler: "
                                        "a ghost penalty needs simple "
                                        "interior knots");
    }

    ElementValues<Basis, n> before(patch_, space_.bases(), penalty.flags());
    ElementValues<Basis, n> after(patch_, space_.bases(), penalty.flags());
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
        const int degree = basis.grid().knots(direction).degree();

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

        face_dofs(cell_before, cell_after, dofs);
        add_to_matrix(stiffness_, dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_stiffness(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2)
{
    add_skeleton_condition_stiffness(condition, quadrature, nullptr, 0,
                                     cell_weights);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_stiffness(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order,
    const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2)
{
    add_skeleton_condition_stiffness(condition, quadrature, &projection, order,
                                     cell_weights);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::add_skeleton_condition_stiffness(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order,
    const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2)
{
    // Without the faces coupled, the pattern would grow with each new face
    if (!couple_faces_)
        throw std::invalid_argument("Assembler: "
                                    "a skeleton condition needs the faces "
                                    "coupled");

    if (cell_weights.size() != space_.num_elements())
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");

    if (projection != nullptr)
        check_projection(*projection, order, quadrature.num_points());

    ElementValues<Basis, n> before(patch_, space_.bases(), condition.flags());
    ElementValues<Basis, n> after(patch_, space_.bases(), condition.flags());
    ElementValues<Basis, n> shifted_before(patch_, space_.bases(),
                                           condition.flags());
    ElementValues<Basis, n> shifted_after(patch_, space_.bases(),
                                          condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;
    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

    for (int face = 0; face < quadrature.num_faces(); ++face) {
        const int cell_before = quadrature.cells()(face, 0);
        const int cell_after = quadrature.cells()(face, 1);
        const std::array<Scalar, 2> side_weights{cell_weights(cell_before),
                                                 cell_weights(cell_after)};

        // Without a jump of the test functions, the condition adds nothing
        if (side_weights[0] == side_weights[1])
            continue;

        const int first = quadrature.offsets()(face);
        const int count = quadrature.offsets()(face + 1) - first;

        // The values of each cell, with its own polynomial, at the same
        // points, and with a projection also expanded towards the boundary
        // along the same distance vectors
        points = quadrature.points().middleRows(first, count);
        before.reinit(cell_before, points);
        after.reinit(cell_after, points);

        if (projection != nullptr) {
            const Eigen::MatrixX<Scalar> distances =
                projection->distances.middleRows(first, count);

            shifted_before.reinit(cell_before, points);
            shifted_after.reinit(cell_after, points);
            shift_values(shifted_before, distances, order);
            shift_values(shifted_after, distances, order);
        }

        const bool shifted = projection != nullptr;

        // Face weights and physical normals through the map, which is
        // continuous across the face, from the cell before to the cell after
        normals = quadrature.normals().middleRows(first, count);
        Patch<Basis, n>::boundary_measure_on_element(before.tangents(),
                                                     normals, measures);
        Patch<Basis, n>::physical_normal_on_element(before.tangents(),
                                                    normals,
                                                    physical_normals);
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(measures);

        condition.local_stiffness(before, after,
                                  shifted ? shifted_before : before,
                                  shifted ? shifted_after : after,
                                  side_weights, weights, physical_normals,
                                  local);

        face_dofs(cell_before, cell_after, dofs);
        add_to_matrix(stiffness_, dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_load(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2)
{
    add_skeleton_condition_load(condition, quadrature, nullptr, 0, data,
                                cell_weights);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_load(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2)
{
    add_skeleton_condition_load(condition, quadrature, &projection, order, data,
                                cell_weights);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::add_skeleton_condition_load(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2)
{
    if (data.size() != quadrature.num_points())
        throw std::invalid_argument("Assembler: "
                                    "the data must have one value per point");

    // Without the faces coupled, the pattern would grow with each new face
    if (!couple_faces_)
        throw std::invalid_argument("Assembler: "
                                    "a skeleton condition needs the faces "
                                    "coupled");

    if (cell_weights.size() != space_.num_elements())
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");

    if (projection != nullptr)
        check_projection(*projection, order, quadrature.num_points());

    ElementValues<Basis, n> before(patch_, space_.bases(), condition.flags());
    ElementValues<Basis, n> after(patch_, space_.bases(), condition.flags());
    ElementValues<Basis, n> shifted_before(patch_, space_.bases(),
                                           condition.flags());
    ElementValues<Basis, n> shifted_after(patch_, space_.bases(),
                                          condition.flags());
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> local;
    std::vector<int> dofs;

    for (int face = 0; face < quadrature.num_faces(); ++face) {
        const int cell_before = quadrature.cells()(face, 0);
        const int cell_after = quadrature.cells()(face, 1);
        const std::array<Scalar, 2> side_weights{cell_weights(cell_before),
                                                 cell_weights(cell_after)};

        // Without a jump of the test functions, the condition adds nothing
        if (side_weights[0] == side_weights[1])
            continue;

        const int first = quadrature.offsets()(face);
        const int count = quadrature.offsets()(face + 1) - first;

        // The values of each cell, with its own polynomial, at the same
        // points, and with a projection also expanded towards the boundary
        // along the same distance vectors
        points = quadrature.points().middleRows(first, count);
        before.reinit(cell_before, points);
        after.reinit(cell_after, points);

        if (projection != nullptr) {
            const Eigen::MatrixX<Scalar> distances =
                projection->distances.middleRows(first, count);

            shifted_before.reinit(cell_before, points);
            shifted_after.reinit(cell_after, points);
            shift_values(shifted_before, distances, order);
            shift_values(shifted_after, distances, order);
        }

        const bool shifted = projection != nullptr;

        // Face weights and physical normals through the map, which is
        // continuous across the face, from the cell before to the cell after
        normals = quadrature.normals().middleRows(first, count);
        Patch<Basis, n>::boundary_measure_on_element(before.tangents(),
                                                     normals, measures);
        Patch<Basis, n>::physical_normal_on_element(before.tangents(),
                                                    normals,
                                                    physical_normals);
        weights = quadrature.weights().segment(first, count);
        weights = weights.cwiseProduct(measures);

        imposed = data.segment(first, count);
        condition.local_load(before, after,
                             shifted ? shifted_before : before,
                             shifted ? shifted_after : after, side_weights,
                             weights, physical_normals, imposed, local);

        face_dofs(cell_before, cell_after, dofs);
        add_to_load(dofs, local);
    }

    finalize();
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::add_to_matrix(Mat matrix,
                                        std::span<const int> dofs,
                                        const Eigen::MatrixX<Scalar>& local)
{
    // The k-th degree of freedom pairs with row and column k
    const PetscInt count = static_cast<PetscInt>(dofs.size());

    // PETSc reads an entry for every pair, past the end of a local matrix
    // of another block size
    if (local.rows() != count || local.cols() != count)
        throw std::invalid_argument("Assembler: "
                                    "the local matrix must match the degrees "
                                    "of freedom of the space");

    check_petsc(MatSetValues(matrix, count, dofs.data(), count, dofs.data(),
                             local.data(), ADD_VALUES));
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::add_to_load(std::span<const int> dofs,
                                      const Eigen::VectorX<Scalar>& local)
{
    // The k-th degree of freedom pairs with row k
    const PetscInt count = static_cast<PetscInt>(dofs.size());

    // PETSc reads an entry for every degree of freedom, past the end of a
    // local load of another block size
    if (local.size() != count)
        throw std::invalid_argument("Assembler: "
                                    "the local load must match the degrees "
                                    "of freedom of the space");

    check_petsc(VecSetValues(load_, count, dofs.data(), local.data(),
                             ADD_VALUES));
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::finalize()
{
    for (Mat matrix : {stiffness_, mass_}) {
        check_petsc(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
        check_petsc(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));
    }

    check_petsc(VecAssemblyBegin(load_));
    check_petsc(VecAssemblyEnd(load_));
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::face_dofs(int before, int after,
                                    std::vector<int>& dofs) const
{
    std::vector<int> after_dofs;
    space_.dofs_on_element(before, dofs);
    space_.dofs_on_element(after, after_dofs);

    dofs.insert(dofs.end(), after_dofs.begin(), after_dofs.end());
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::clear()
{
    // Only the stored values, so that the pattern stays
    check_petsc(MatZeroEntries(stiffness_));
    check_petsc(MatZeroEntries(mass_));
    check_petsc(VecZeroEntries(load_));
}

template class Assembler<TensorBSpline<double, 2>, 2>;
template class Assembler<TensorBSpline<double, 3>, 3>;
template class Assembler<TensorNURBS<double, 2>, 2>;
template class Assembler<TensorNURBS<double, 3>, 3>;

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_mass(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_mass(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_mass(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_mass(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_mass(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_mass(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_mass(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_mass(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_mass(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_mass(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_mass(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_mass(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_mass(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_mass(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_mass(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_mass(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_stiffness(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_stiffness(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_load(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_load(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_mass(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_mass(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_mass(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_mass(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_mass(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_mass(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_mass(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_mass(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const NeumannCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const NeumannCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const NeumannCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const NeumannCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_stiffness(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_stiffness(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_load(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&, const Eigen::VectorX<double>&);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_load(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&,
    const BoundaryProjection<double>&, int,
    const Eigen::VectorX<double>&, const Eigen::VectorX<double>&);

} // namespace iguana
