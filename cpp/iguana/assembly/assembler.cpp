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
#include <iterator>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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
 *        the boundary they are projected onto, along their distance vectors,
 *        by the Taylor series or by the shift weights of the projection,
 *        about its shift center
 *
 * @param values Values at the points, which are shifted
 * @param projection Projection of the points of a whole quadrature
 * @param first First point of the element or face in @p projection
 * @param count Number of points of the element or face
 * @param order Highest total order of the derivatives
 *
 * @throws std::invalid_argument If the patch is not a B-spline one, whose
 *         basis alone provides the derivatives
 */
template<typename Basis, std::size_t n>
void shift_values(ElementValues<Basis, n>& values,
                  const BoundaryProjection<typename Basis::Scalar>& projection,
                  int first, int count, int order)
{
    using Scalar = typename Basis::Scalar;
    constexpr std::size_t dim = Basis::dimension;

    if constexpr (std::same_as<Basis, TensorBSpline<Scalar, dim>>) {
        const Eigen::MatrixX<Scalar> distances =
            projection.distances.middleRows(first, count);

        if (projection.shift_weights.size() == 0) {
            values.shift(distances, order);
        }
        else {
            // The further sets of weights, at the points of the element or
            // face alone
            std::vector<Eigen::MatrixX<Scalar>> coupled;

            for (const Eigen::MatrixX<Scalar>& set
                 : projection.coupled_weights)
                coupled.emplace_back(set.middleRows(first, count));

            values.shift(distances,
                         projection.shift_weights.middleRows(first, count),
                         coupled, order, projection.shift_center);
        }
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
 * @throws std::invalid_argument If the order is negative, if the
 *         projection does not have one distance vector per point, if it
 *         has shift weights without one row per point and one column per
 *         multi-index of the order, or further weights without shift
 *         weights or of another size
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

    const Eigen::MatrixX<T>& weights = projection.shift_weights;

    if (weights.size() > 0 && (weights.rows() != num_points
                               || weights.cols() != simplex_size(order)))
        throw std::invalid_argument("Assembler: "
                                    "the shift weights must have one row "
                                    "per point and one column per "
                                    "multi-index of the order");

    for (const Eigen::MatrixX<T>& set : projection.coupled_weights) {
        if (weights.size() == 0 || set.rows() != weights.rows()
            || set.cols() != weights.cols())
            throw std::invalid_argument("Assembler: "
                                        "the coupled weights must have the "
                                        "size of the shift weights");
    }
}

/**
 * @brief Checks that the cell weights of a pass weight every element
 *
 * @throws std::invalid_argument If the cell weights do not have one value
 *         per element
 */
template<std::floating_point T>
void check_cell_weights(const Eigen::VectorX<T>& cell_weights,
                        int num_elements)
{
    if (cell_weights.size() != num_elements)
        throw std::invalid_argument("Assembler: "
                                    "the cell weights must have one value "
                                    "per element");
}

/**
 * @brief Checks the source and the cell weights of an element pass
 *
 * @throws std::invalid_argument If the source does not have one row per
 *         point, or the cell weights one value per element
 */
template<std::floating_point T>
void check_element_data(const Eigen::MatrixX<T>& source,
                        const Eigen::VectorX<T>& cell_weights,
                        int num_points, int num_elements)
{
    if (source.rows() != num_points)
        throw std::invalid_argument("Assembler: "
                                    "the source must have one row per point");

    check_cell_weights(cell_weights, num_elements);
}

/**
 * @brief Checks the data and the projection, if any, of a condition pass
 *
 * @throws std::invalid_argument If the data does not have one value per
 *         point, or as check_projection() does
 */
template<std::floating_point T>
void check_condition_data(const Eigen::VectorX<T>& data,
                          const BoundaryProjection<T>* projection, int order,
                          int num_points)
{
    if (data.size() != num_points)
        throw std::invalid_argument("Assembler: "
                                    "the data must have one value per point");

    if (projection != nullptr)
        check_projection(*projection, order, num_points);
}

/**
 * @brief Checks the faces, the data, the projection, if any, and the cell
 *        weights of a skeleton condition pass
 *
 * @throws std::invalid_argument If the faces are not coupled, or as
 *         check_condition_data() and check_cell_weights() do
 */
template<std::floating_point T>
void check_skeleton_data(bool couple_faces, const Eigen::VectorX<T>& data,
                         const BoundaryProjection<T>* projection, int order,
                         const Eigen::VectorX<T>& cell_weights,
                         int num_points, int num_elements)
{
    // Without the faces coupled, the pattern would grow with each new face
    if (!couple_faces)
        throw std::invalid_argument("Assembler: "
                                    "a skeleton condition needs the faces "
                                    "coupled");

    check_condition_data(data, projection, order, num_points);
    check_cell_weights(cell_weights, num_elements);
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

/**
 * @brief Sums the rows and columns of the local matrix of a face that share
 *        a degree of freedom, those of a function active on both of its
 *        cells, so that PETSc adds each entry once
 *
 * @param local Local matrix, with a row and a column per row of the face
 * @param positions Index of the degree of freedom of each row of the face
 *        among those of the face
 * @param num_dofs Number of degrees of freedom of the face
 * @param folded Output, overwritten, with a row and a column per degree of
 *        freedom of the face
 *
 * @throws std::invalid_argument If @p local does not have a row and a
 *         column per entry of @p positions
 */
template<std::floating_point T>
void fold_face(const Eigen::MatrixX<T>& local,
               const std::vector<int>& positions, std::ptrdiff_t num_dofs,
               Eigen::MatrixX<T>& folded)
{
    const std::ptrdiff_t count = std::ssize(positions);

    // PETSc would otherwise read past the end of a local matrix of another
    // block size, and so would the sums
    if (local.rows() != count || local.cols() != count)
        throw std::invalid_argument("Assembler: "
                                    "the local matrix must match the degrees "
                                    "of freedom of the space");

    folded.setZero(num_dofs, num_dofs);

    for (std::ptrdiff_t col = 0; col < count; ++col) {
        for (std::ptrdiff_t row = 0; row < count; ++row)
            folded(positions[row], positions[col]) += local(row, col);
    }
}

/**
 * @brief Sums the entries of the local residual of a face that share a
 *        degree of freedom, as for a local matrix
 *
 * @throws std::invalid_argument If @p local does not have an entry per
 *         entry of @p positions
 */
template<std::floating_point T>
void fold_face(const Eigen::VectorX<T>& local,
               const std::vector<int>& positions, std::ptrdiff_t num_dofs,
               Eigen::VectorX<T>& folded)
{
    const std::ptrdiff_t count = std::ssize(positions);

    if (local.size() != count)
        throw std::invalid_argument("Assembler: "
                                    "the local residual must match the "
                                    "degrees of freedom of the space");

    folded.setZero(num_dofs);

    for (std::ptrdiff_t row = 0; row < count; ++row)
        folded(positions[row]) += local(row);
}

/**
 * @brief Cells around an element of a grid: the element itself and, with
 *        the faces coupled, its neighbours across each of its faces
 *
 * @param element Element index, in the numbering of the grid
 * @param counts Number of elements of the grid in each direction
 * @param couple_faces Whether to add the neighbours across the faces
 * @param cells Output, overwritten
 */
template<std::size_t d>
void cells_around(int element, const std::array<int, d>& counts,
                  bool couple_faces, std::vector<int>& cells)
{
    cells.assign(1, element);

    if (!couple_faces)
        return;

    const std::array<int, d> index = unflatten(element, counts);

    for (std::size_t direction = 0; direction < d; ++direction) {
        for (const int step : {-1, 1}) {
            std::array<int, d> neighbour = index;
            neighbour[direction] += step;

            if (neighbour[direction] >= 0
                && neighbour[direction] < counts[direction])
                cells.push_back(flatten(neighbour, counts));
        }
    }
}

/**
 * @brief Compressed rows of the sparsity pattern of a space: an entry for
 *        every pair of degrees of freedom sharing an element and, with the
 *        faces coupled, for every pair on the two cells of an interior face
 *
 * Each row takes the degrees of freedom of the cells around its elements
 * once, skipping those it already holds, so that building the pattern
 * takes the memory of the pattern itself
 *
 * @param space Space whose degrees of freedom number the rows and columns
 * @param counts Number of elements of the grid in each direction
 * @param couple_faces Whether to pair the two cells of every interior face
 * @param offsets Output, the first entry of each row, followed by the
 *        number of entries
 * @param columns Output, the columns of each row in increasing order
 */
template<typename Basis>
void sparsity_pattern(const MixedFunctionSpace<Basis>& space,
                      const std::array<int, Basis::dimension>& counts,
                      bool couple_faces, std::vector<PetscInt>& offsets,
                      std::vector<PetscInt>& columns)
{
    const int num_elements = space.num_elements();
    const int num_dofs = space.num_dofs();

    // The degrees of freedom of each element, and the elements of each
    // degree of freedom, outside cells listing none
    std::vector<std::vector<int>> element_dofs(num_elements);
    std::vector<std::vector<int>> dof_elements(num_dofs);

    for (int element = 0; element < num_elements; ++element) {
        space.dofs_on_element(element, element_dofs[element]);

        for (const int dof : element_dofs[element])
            dof_elements[dof].push_back(element);
    }

    // The last row that took each column, which marks the columns a row
    // already holds without clearing anything between rows
    std::vector<int> last_row(num_dofs, -1);
    std::vector<int> cells;
    offsets.assign(1, 0);
    columns.clear();

    for (int row = 0; row < num_dofs; ++row) {
        for (const int element : dof_elements[row]) {
            cells_around(element, counts, couple_faces, cells);

            for (const int cell : cells) {
                for (const int col : element_dofs[cell]) {
                    if (last_row[col] == row)
                        continue;

                    last_row[col] = row;
                    columns.push_back(col);
                }
            }
        }

        std::sort(columns.begin() + offsets.back(), columns.end());
        offsets.push_back(static_cast<PetscInt>(columns.size()));
    }
}

/**
 * @brief Visits the cells of a domain quadrature whose weight is not zero,
 *        with the values at their points and the physical weights there
 *
 * @details The physical weights are the quadrature weights times the
 *          measure, times the weight of the test functions of the cell,
 *          constant on it
 *
 * @param visit Called with the cell, the first of its points and their
 *        number, the values and the weights
 */
template<typename Basis, std::size_t n, typename Visit>
void visit_cells(
    const Patch<Basis, n>& patch, const MixedFunctionSpace<Basis>& space,
    ValueFlags flags,
    const QuadratureDomain<typename Basis::Scalar, Basis::dimension>&
        quadrature,
    const Eigen::VectorX<typename Basis::Scalar>& cell_weights, Visit&& visit)
{
    using Scalar = typename Basis::Scalar;

    ElementValues<Basis, n> values(patch, space.bases(), flags);
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights;

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

        visit(cell, first, count, values, weights);
    }
}

/**
 * @brief Visits the cells of a boundary quadrature, with the values at
 *        their points, expanded along a projection when one is given, and
 *        the boundary weights and physical normals there
 *
 * @param visit Called with the cell, the first of its points and their
 *        number, the values, the shifted values, the weights and the
 *        normals
 */
template<typename Basis, std::size_t n, typename Visit>
void visit_boundary_cells(
    const Patch<Basis, n>& patch, const MixedFunctionSpace<Basis>& space,
    ValueFlags flags,
    const QuadratureBoundary<typename Basis::Scalar, Basis::dimension>&
        quadrature,
    const BoundaryProjection<typename Basis::Scalar>* projection, int order,
    Visit&& visit)
{
    using Scalar = typename Basis::Scalar;

    ElementValues<Basis, n> values(patch, space.bases(), flags);
    ElementValues<Basis, n> shifted_values(patch, space.bases(), flags);
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;

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
            shift_values(shifted_values, *projection, first, count, order);
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

        visit(cell, first, count, values, shifted, weights, physical_normals);
    }
}

/**
 * @brief Visits the faces of a skeleton quadrature across which the
 *        weights of the cells jump, with the values of each cell at their
 *        points, expanded along a projection when one is given, and the
 *        face weights and physical normals there
 *
 * @param visit Called with the cells before and after the face, the first
 *        of its points and their number, the values of each cell, their
 *        shifted values, the weights of the two cells, the face weights and
 *        the normals
 */
template<typename Basis, std::size_t n, typename Visit>
void visit_jump_faces(
    const Patch<Basis, n>& patch, const MixedFunctionSpace<Basis>& space,
    ValueFlags flags,
    const QuadratureSkeleton<typename Basis::Scalar, Basis::dimension>&
        quadrature,
    const BoundaryProjection<typename Basis::Scalar>* projection, int order,
    const Eigen::VectorX<typename Basis::Scalar>& cell_weights, Visit&& visit)
{
    using Scalar = typename Basis::Scalar;

    ElementValues<Basis, n> before(patch, space.bases(), flags);
    ElementValues<Basis, n> after(patch, space.bases(), flags);
    ElementValues<Basis, n> shifted_before(patch, space.bases(), flags);
    ElementValues<Basis, n> shifted_after(patch, space.bases(), flags);
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;
    PointMatrix<Scalar, n> physical_normals;

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
            shifted_before.reinit(cell_before, points);
            shifted_after.reinit(cell_after, points);
            shift_values(shifted_before, *projection, first, count, order);
            shift_values(shifted_after, *projection, first, count, order);
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

        visit(cell_before, cell_after, first, count, before, after,
              shifted ? shifted_before : before,
              shifted ? shifted_after : after, side_weights, weights,
              physical_normals);
    }
}

/**
 * @brief Visits the faces of a skeleton quadrature, with the values of each
 *        cell at their points differentiated across the face to the degree
 *        of a basis there, and the face weights
 *
 * @param visit Called with the cells before and after the face, the values
 *        of each, the weights and the degree
 */
template<typename Basis, std::size_t n, typename Visit>
void visit_ghost_faces(
    const Patch<Basis, n>& patch, const MixedFunctionSpace<Basis>& space,
    ValueFlags flags,
    const QuadratureSkeleton<typename Basis::Scalar, Basis::dimension>&
        quadrature,
    const Basis& basis, Visit&& visit)
{
    using Scalar = typename Basis::Scalar;

    ElementValues<Basis, n> before(patch, space.bases(), flags);
    ElementValues<Basis, n> after(patch, space.bases(), flags);
    Eigen::MatrixX<Scalar> points;
    Eigen::MatrixX<Scalar> normals;
    Eigen::VectorX<Scalar> measures;
    Eigen::VectorX<Scalar> weights;

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

        visit(cell_before, cell_after, before, after, weights, degree);
    }
}

/**
 * @brief Visits the faces of a skeleton quadrature with the values of both
 *        of their cells at the points a domain quadrature holds on each,
 *        every cell with its own polynomial, extended into the other, and
 *        the volume weights of each cell
 *
 * @param visit Called with the cells before and after the face, the values
 *        of the cell before and of the cell after at the points of the cell
 *        before, those of the cell before and of the cell after at the
 *        points of the cell after, and the weights of each cell
 *
 * @throws std::invalid_argument If a cell of a face holds no points of the
 *         domain quadrature
 */
template<typename Basis, std::size_t n, typename Visit>
void visit_ghost_patches(
    const Patch<Basis, n>& patch, const MixedFunctionSpace<Basis>& space,
    ValueFlags flags,
    const QuadratureSkeleton<typename Basis::Scalar, Basis::dimension>& faces,
    const QuadratureDomain<typename Basis::Scalar, Basis::dimension>& cells,
    Visit&& visit)
{
    using Scalar = typename Basis::Scalar;

    // The position of each cell among those the domain quadrature holds,
    // -1 for none
    std::vector<int> held(static_cast<std::size_t>(space.num_elements()), -1);

    for (int index = 0; index < cells.num_elements(); ++index)
        held[static_cast<std::size_t>(cells.elements()(index))] = index;

    ElementValues<Basis, n> before(patch, space.bases(), flags);
    ElementValues<Basis, n> after_in_before(patch, space.bases(), flags);
    ElementValues<Basis, n> before_in_after(patch, space.bases(), flags);
    ElementValues<Basis, n> after(patch, space.bases(), flags);
    Eigen::MatrixX<Scalar> points;
    Eigen::VectorX<Scalar> weights_before;
    Eigen::VectorX<Scalar> weights_after;

    // The points and weights the domain quadrature holds on a cell
    const auto rule = [&cells, &held](int cell, Eigen::MatrixX<Scalar>& at,
                                      Eigen::VectorX<Scalar>& weights) {
        const int index = held[static_cast<std::size_t>(cell)];

        if (index < 0)
            throw std::invalid_argument("Assembler: "
                                        "a volume ghost penalty needs points "
                                        "on the cells of its faces");

        const int first = cells.offsets()(index);
        const int count = cells.offsets()(index + 1) - first;
        at = cells.points().middleRows(first, count);
        weights = cells.weights().segment(first, count);
    };

    for (int face = 0; face < faces.num_faces(); ++face) {
        const int cell_before = faces.cells()(face, 0);
        const int cell_after = faces.cells()(face, 1);

        // At the points of each cell, its own values and those of the other
        // cell, whose polynomial an evaluation on it extends, and the volume
        // weights of the cell through the map
        rule(cell_before, points, weights_before);
        before.reinit(cell_before, points);
        after_in_before.reinit(cell_after, points);
        weights_before = weights_before.cwiseProduct(before.measures());

        rule(cell_after, points, weights_after);
        before_in_after.reinit(cell_before, points);
        after.reinit(cell_after, points);
        weights_after = weights_after.cwiseProduct(after.measures());

        visit(cell_before, cell_after, before, after_in_before,
              before_in_after, after, weights_before, weights_after);
    }
}

/**
 * @brief Read access to the entries of a PETSc vector for the length of a
 *        pass, given back even when the pass throws
 */
class Entries
{
public:
    explicit Entries(Vec vector) : vector_(vector)
    {
        check_petsc(VecGetArrayRead(vector_, &entries_));
    }

    ~Entries()
    {
        // A destructor must not throw, so a failure is left to PETSc to
        // report
        static_cast<void>(VecRestoreArrayRead(vector_, &entries_));
    }

    Entries(const Entries&) = delete;
    Entries& operator=(const Entries&) = delete;

    /// @brief The entries at degrees of freedom, one per row
    template<std::floating_point T>
    void gather(std::span<const int> dofs, Eigen::VectorX<T>& local) const
    {
        local.resize(std::ssize(dofs));

        for (std::ptrdiff_t row = 0; row < local.size(); ++row)
            local(row) = entries_[dofs[row]];
    }

    /// @brief The entries at the degrees of freedom of the rows of a face,
    ///        given by their positions among the degrees of freedom
    template<std::floating_point T>
    void gather(std::span<const int> dofs, std::span<const int> positions,
                Eigen::VectorX<T>& local) const
    {
        local.resize(std::ssize(positions));

        for (std::ptrdiff_t row = 0; row < local.size(); ++row)
            local(row) = entries_[dofs[positions[row]]];
    }

private:
    /// @brief Vector whose entries are read
    Vec vector_;

    /// @brief Its entries, while they are read
    const PetscScalar* entries_ = nullptr;
};

/**
 * @brief Basis of the field a ghost penalty extends, checked to suit it
 *
 * @throws std::invalid_argument If the faces are not coupled, if the map of
 *         the patch is not affine, if the field of the penalty is not one of
 *         the space, or if a knot vector of that field repeats an interior
 *         knot
 */
template<typename Basis, std::size_t n>
const Basis& ghost_basis(bool couple_faces, const Patch<Basis, n>& patch,
                         const MixedFunctionSpace<Basis>& space,
                         const SkeletonGhostPenalty<Basis, n>& penalty)
{
    // Without the faces coupled, the pattern would grow with each new face
    if (!couple_faces)
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty needs the faces coupled");

    // Only on an affine map do the jumps of the derivatives across the knot
    // lines give those of the physical normal derivatives
    if (!patch.is_affine())
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty needs an affine patch");

    // The penalty extends one field, whose basis gives the degrees
    const std::size_t field = penalty.field(space.num_fields());

    if (field >= space.num_fields())
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty must act on a field of "
                                    "the space");

    const Basis& basis = space.field(field).basis();

    // A repeated knot of the field would let lower derivatives of its
    // functions jump unpenalized
    for (std::size_t direction = 0; direction < Basis::dimension;
         ++direction) {
        if (!simple_interior_knots(basis.grid().knots(direction)))
            throw std::invalid_argument("Assembler: "
                                        "a ghost penalty needs simple "
                                        "interior knots");
    }

    return basis;
}

/**
 * @brief Checks that a volume ghost penalty suits the assembly
 *
 * @throws std::invalid_argument If the faces are not coupled, or if the
 *         field of the penalty is not one of the space
 */
template<typename Basis, std::size_t n>
void check_volume_ghost(bool couple_faces,
                        const MixedFunctionSpace<Basis>& space,
                        const VolumeGhostPenalty<Basis, n>& penalty)
{
    // Without the faces coupled, the pattern would grow with each new face
    if (!couple_faces)
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty needs the faces coupled");

    if (penalty.field(space.num_fields()) >= space.num_fields())
        throw std::invalid_argument("Assembler: "
                                    "a ghost penalty must act on a field of "
                                    "the space");
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

    const int num_dofs = space.num_dofs();

    // Sparsity pattern: K_ij can only be non-zero when functions i and j
    // share an element, and a term on a face pairs the functions of its two
    // cells, so with the faces coupled, so do the cells of every interior
    // face of the grid
    std::array<int, dim> counts{};

    for (std::size_t direction = 0; direction < dim; ++direction)
        counts[direction] =
            patch.basis().grid().knots(direction).num_elements();

    std::vector<PetscInt> offsets;
    std::vector<PetscInt> columns;
    sparsity_pattern(space, counts, couple_faces, offsets, columns);

    // The compressed rows of the pattern preallocate the tangent, its
    // explicit zeros the only entries a write may reach. The block size
    // tells the solvers which degrees of freedom are components of one
    // function, in blocks only when the space has a single field
    const int block_size =
        space.num_fields() == 1 ? space.field(0).block_size() : 1;

    check_petsc(MatCreate(PETSC_COMM_SELF, &tangent_));
    check_petsc(MatSetSizes(tangent_, num_dofs, num_dofs, num_dofs,
                            num_dofs));
    check_petsc(MatSetBlockSize(tangent_, block_size));
    check_petsc(MatSetType(tangent_, MATSEQAIJ));
    check_petsc(MatSeqAIJSetPreallocationCSR(tangent_, offsets.data(),
                                             columns.data(), nullptr));

    // The mass shares the pattern, and both read local matrices by columns,
    // as Eigen stores them, and refuse entries outside the pattern
    check_petsc(MatDuplicate(tangent_, MAT_SHARE_NONZERO_PATTERN, &mass_));

    for (Mat matrix : {tangent_, mass_}) {
        check_petsc(MatSetOption(matrix, MAT_ROW_ORIENTED, PETSC_FALSE));
        check_petsc(MatSetOption(matrix, MAT_NEW_NONZERO_ALLOCATION_ERR,
                                 PETSC_TRUE));
    }

    // The residual follows the rows, with their block size
    check_petsc(MatCreateVecs(tangent_, nullptr, &residual_));
    check_petsc(VecZeroEntries(residual_));
}

template<typename Basis, std::size_t n>
Assembler<Basis, n>::~Assembler()
{
    // A destructor must not throw, so a failure is left to PETSc to report
    static_cast<void>(MatDestroy(&mass_));
    static_cast<void>(MatDestroy(&tangent_));
    static_cast<void>(VecDestroy(&residual_));
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_tangent(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::MatrixX<Scalar>& source, Vec coefficients)
{
    // Every cell weighted by one
    const Eigen::VectorX<Scalar> ones =
        Eigen::VectorX<Scalar>::Ones(space_.num_elements());

    assemble_tangent(element, quadrature, source, ones, coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_tangent(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
{
    check_element_data(source, cell_weights, quadrature.num_points(),
                       space_.num_elements());
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::MatrixX<Scalar> sources;
    Eigen::VectorX<Scalar> cell_coefficients;
    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

    visit_cells(patch_, space_, element.flags(), quadrature, cell_weights,
                [&](int cell, int first, int count, const auto& values,
                    const auto& weights) {
                    space_.dofs_on_element(cell, dofs);
                    entries.gather(dofs, cell_coefficients);
                    sources = source.middleRows(first, count);
                    element.local_tangent(values, weights, sources,
                                          cell_coefficients, local);
                    add_to_matrix(tangent_, dofs, local);
                });

    finalize(tangent_);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_residual(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::MatrixX<Scalar>& source, Vec coefficients)
{
    // Every cell weighted by one
    const Eigen::VectorX<Scalar> ones =
        Eigen::VectorX<Scalar>::Ones(space_.num_elements());

    assemble_residual(element, quadrature, source, ones, coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Element<Basis, n>> E>
void Assembler<Basis, n>::assemble_residual(
    const E& element, const QuadratureDomain<Scalar, dim>& quadrature,
    const Eigen::MatrixX<Scalar>& source,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
{
    check_element_data(source, cell_weights, quadrature.num_points(),
                       space_.num_elements());
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::MatrixX<Scalar> sources;
    Eigen::VectorX<Scalar> cell_coefficients;
    Eigen::VectorX<Scalar> local;
    std::vector<int> dofs;

    visit_cells(patch_, space_, element.flags(), quadrature, cell_weights,
                [&](int cell, int first, int count, const auto& values,
                    const auto& weights) {
                    space_.dofs_on_element(cell, dofs);
                    entries.gather(dofs, cell_coefficients);
                    sources = source.middleRows(first, count);
                    element.local_residual(values, weights, sources,
                                           cell_coefficients, local);
                    add_to_residual(dofs, local);
                });

    finalize(residual_);
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
    check_cell_weights(cell_weights, space_.num_elements());

    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

    visit_cells(patch_, space_, element.flags(), quadrature, cell_weights,
                [&](int cell, int, int, const auto& values,
                    const auto& weights) {
                    element.local_mass(values, weights, local);
                    space_.dofs_on_element(cell, dofs);
                    add_to_matrix(mass_, dofs, local);
                });

    finalize(mass_);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_tangent(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& data, Vec coefficients)
{
    add_condition_tangent(condition, quadrature, nullptr, 0, data,
                          coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_residual(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& data, Vec coefficients)
{
    add_condition_residual(condition, quadrature, nullptr, 0, data,
                           coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_tangent(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order,
    const Eigen::VectorX<Scalar>& data, Vec coefficients) requires (dim == 2)
{
    add_condition_tangent(condition, quadrature, &projection, order, data,
                          coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::assemble_residual(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order,
    const Eigen::VectorX<Scalar>& data, Vec coefficients) requires (dim == 2)
{
    add_condition_residual(condition, quadrature, &projection, order, data,
                           coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::add_condition_tangent(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order,
    const Eigen::VectorX<Scalar>& data, Vec coefficients)
{
    check_condition_data(data, projection, order, quadrature.num_points());
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> cell_coefficients;
    Eigen::MatrixX<Scalar> local;
    std::vector<int> dofs;

    visit_boundary_cells(
        patch_, space_, condition.flags(), quadrature, projection, order,
        [&](int cell, int first, int count, const auto& values,
            const auto& shifted, const auto& weights, const auto& normals) {
            space_.dofs_on_element(cell, dofs);
            entries.gather(dofs, cell_coefficients);
            imposed = data.segment(first, count);
            condition.local_tangent(values, shifted, weights, normals,
                                    imposed, cell_coefficients, local);
            add_to_matrix(tangent_, dofs, local);
        });

    finalize(tangent_);
}

template<typename Basis, std::size_t n>
template<std::derived_from<Condition<Basis, n>> C>
void Assembler<Basis, n>::add_condition_residual(
    const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order,
    const Eigen::VectorX<Scalar>& data, Vec coefficients)
{
    check_condition_data(data, projection, order, quadrature.num_points());
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> cell_coefficients;
    Eigen::VectorX<Scalar> local;
    std::vector<int> dofs;

    visit_boundary_cells(
        patch_, space_, condition.flags(), quadrature, projection, order,
        [&](int cell, int first, int count, const auto& values,
            const auto& shifted, const auto& weights, const auto& normals) {
            space_.dofs_on_element(cell, dofs);
            entries.gather(dofs, cell_coefficients);
            imposed = data.segment(first, count);
            condition.local_residual(values, shifted, weights, normals,
                                     imposed, cell_coefficients, local);
            add_to_residual(dofs, local);
        });

    finalize(residual_);
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::assemble_tangent(
    const SkeletonGhostPenalty<Basis, n>& penalty,
    const QuadratureSkeleton<Scalar, dim>& quadrature, Vec coefficients)
    requires (dim == 2)
{
    const Basis& basis = ghost_basis(couple_faces_, patch_, space_, penalty);
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> face_coefficients;
    Eigen::MatrixX<Scalar> local;
    Eigen::MatrixX<Scalar> folded;
    std::vector<int> dofs;
    std::vector<int> positions;

    visit_ghost_faces(
        patch_, space_, penalty.flags(), quadrature, basis,
        [&](int cell_before, int cell_after, const auto& before,
            const auto& after, const auto& weights, int degree) {
            face_dofs(cell_before, cell_after, dofs, positions);
            entries.gather(dofs, positions, face_coefficients);
            penalty.local_tangent(before, after, weights, degree,
                                  face_coefficients, local);

            fold_face(local, positions, std::ssize(dofs), folded);
            add_to_matrix(tangent_, dofs, folded);
        });

    finalize(tangent_);
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::assemble_residual(
    const SkeletonGhostPenalty<Basis, n>& penalty,
    const QuadratureSkeleton<Scalar, dim>& quadrature, Vec coefficients)
    requires (dim == 2)
{
    const Basis& basis = ghost_basis(couple_faces_, patch_, space_, penalty);
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> face_coefficients;
    Eigen::VectorX<Scalar> local;
    Eigen::VectorX<Scalar> folded;
    std::vector<int> dofs;
    std::vector<int> positions;

    visit_ghost_faces(
        patch_, space_, penalty.flags(), quadrature, basis,
        [&](int cell_before, int cell_after, const auto& before,
            const auto& after, const auto& weights, int degree) {
            face_dofs(cell_before, cell_after, dofs, positions);
            entries.gather(dofs, positions, face_coefficients);
            penalty.local_residual(before, after, weights, degree,
                                   face_coefficients, local);

            fold_face(local, positions, std::ssize(dofs), folded);
            add_to_residual(dofs, folded);
        });

    finalize(residual_);
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::assemble_tangent(
    const VolumeGhostPenalty<Basis, n>& penalty,
    const QuadratureSkeleton<Scalar, dim>& faces,
    const QuadratureDomain<Scalar, dim>& cells, Vec coefficients)
    requires (dim == 2)
{
    check_volume_ghost(couple_faces_, space_, penalty);
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> face_coefficients;
    Eigen::MatrixX<Scalar> local;
    Eigen::MatrixX<Scalar> folded;
    std::vector<int> dofs;
    std::vector<int> positions;

    visit_ghost_patches(
        patch_, space_, penalty.flags(), faces, cells,
        [&](int cell_before, int cell_after, const auto& before,
            const auto& after_in_before, const auto& before_in_after,
            const auto& after, const auto& weights_before,
            const auto& weights_after) {
            face_dofs(cell_before, cell_after, dofs, positions);
            entries.gather(dofs, positions, face_coefficients);
            penalty.local_tangent(before, after_in_before, before_in_after,
                                  after, weights_before, weights_after,
                                  face_coefficients, local);

            fold_face(local, positions, std::ssize(dofs), folded);
            add_to_matrix(tangent_, dofs, folded);
        });

    finalize(tangent_);
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::assemble_residual(
    const VolumeGhostPenalty<Basis, n>& penalty,
    const QuadratureSkeleton<Scalar, dim>& faces,
    const QuadratureDomain<Scalar, dim>& cells, Vec coefficients)
    requires (dim == 2)
{
    check_volume_ghost(couple_faces_, space_, penalty);
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> face_coefficients;
    Eigen::VectorX<Scalar> local;
    Eigen::VectorX<Scalar> folded;
    std::vector<int> dofs;
    std::vector<int> positions;

    visit_ghost_patches(
        patch_, space_, penalty.flags(), faces, cells,
        [&](int cell_before, int cell_after, const auto& before,
            const auto& after_in_before, const auto& before_in_after,
            const auto& after, const auto& weights_before,
            const auto& weights_after) {
            face_dofs(cell_before, cell_after, dofs, positions);
            entries.gather(dofs, positions, face_coefficients);
            penalty.local_residual(before, after_in_before, before_in_after,
                                   after, weights_before, weights_after,
                                   face_coefficients, local);

            fold_face(local, positions, std::ssize(dofs), folded);
            add_to_residual(dofs, folded);
        });

    finalize(residual_);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_tangent(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
    requires (dim == 2)
{
    add_skeleton_condition_tangent(condition, quadrature, nullptr, 0, data,
                                   cell_weights, coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_residual(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
    requires (dim == 2)
{
    add_skeleton_condition_residual(condition, quadrature, nullptr, 0, data,
                                    cell_weights, coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_tangent(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
    requires (dim == 2)
{
    add_skeleton_condition_tangent(condition, quadrature, &projection, order,
                                   data, cell_weights, coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::assemble_residual(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>& projection, int order,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
    requires (dim == 2)
{
    add_skeleton_condition_residual(condition, quadrature, &projection, order,
                                    data, cell_weights, coefficients);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::add_skeleton_condition_tangent(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
    requires (dim == 2)
{
    check_skeleton_data(couple_faces_, data, projection, order, cell_weights,
                        quadrature.num_points(), space_.num_elements());
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> face_coefficients;
    Eigen::MatrixX<Scalar> local;
    Eigen::MatrixX<Scalar> folded;
    std::vector<int> dofs;
    std::vector<int> positions;

    visit_jump_faces(
        patch_, space_, condition.flags(), quadrature, projection, order,
        cell_weights,
        [&](int cell_before, int cell_after, int first, int count,
            const auto& before, const auto& after, const auto& shifted_before,
            const auto& shifted_after, const auto& side_weights,
            const auto& weights, const auto& normals) {
            face_dofs(cell_before, cell_after, dofs, positions);
            entries.gather(dofs, positions, face_coefficients);
            imposed = data.segment(first, count);
            condition.local_tangent(before, after, shifted_before,
                                    shifted_after, side_weights, weights,
                                    normals, imposed, face_coefficients,
                                    local);

            fold_face(local, positions, std::ssize(dofs), folded);
            add_to_matrix(tangent_, dofs, folded);
        });

    finalize(tangent_);
}

template<typename Basis, std::size_t n>
template<std::derived_from<SkeletonCondition<Basis, n>> C>
void Assembler<Basis, n>::add_skeleton_condition_residual(
    const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
    const BoundaryProjection<Scalar>* projection, int order,
    const Eigen::VectorX<Scalar>& data,
    const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
    requires (dim == 2)
{
    check_skeleton_data(couple_faces_, data, projection, order, cell_weights,
                        quadrature.num_points(), space_.num_elements());
    check_coefficients(coefficients);

    const Entries entries(coefficients);
    Eigen::VectorX<Scalar> imposed;
    Eigen::VectorX<Scalar> face_coefficients;
    Eigen::VectorX<Scalar> local;
    Eigen::VectorX<Scalar> folded;
    std::vector<int> dofs;
    std::vector<int> positions;

    visit_jump_faces(
        patch_, space_, condition.flags(), quadrature, projection, order,
        cell_weights,
        [&](int cell_before, int cell_after, int first, int count,
            const auto& before, const auto& after, const auto& shifted_before,
            const auto& shifted_after, const auto& side_weights,
            const auto& weights, const auto& normals) {
            face_dofs(cell_before, cell_after, dofs, positions);
            entries.gather(dofs, positions, face_coefficients);
            imposed = data.segment(first, count);
            condition.local_residual(before, after, shifted_before,
                                     shifted_after, side_weights, weights,
                                     normals, imposed, face_coefficients,
                                     local);

            fold_face(local, positions, std::ssize(dofs), folded);
            add_to_residual(dofs, folded);
        });

    finalize(residual_);
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::check_coefficients(Vec coefficients) const
{
    PetscInt size = 0;
    check_petsc(VecGetSize(coefficients, &size));

    if (size != space_.num_dofs())
        throw std::invalid_argument("Assembler: "
                                    "the coefficients must have one entry "
                                    "per degree of freedom");
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
void Assembler<Basis, n>::add_to_residual(std::span<const int> dofs,
                                          const Eigen::VectorX<Scalar>& local)
{
    // The k-th degree of freedom pairs with row k
    const PetscInt count = static_cast<PetscInt>(dofs.size());

    // PETSc reads an entry for every degree of freedom, past the end of a
    // local residual of another block size
    if (local.size() != count)
        throw std::invalid_argument("Assembler: "
                                    "the local residual must match the "
                                    "degrees of freedom of the space");

    check_petsc(VecSetValues(residual_, count, dofs.data(), local.data(),
                             ADD_VALUES));
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::finalize(Mat matrix)
{
    check_petsc(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
    check_petsc(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::finalize(Vec vector)
{
    check_petsc(VecAssemblyBegin(vector));
    check_petsc(VecAssemblyEnd(vector));
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::face_dofs(int before, int after,
                                    std::vector<int>& dofs,
                                    std::vector<int>& positions) const
{
    std::vector<int> after_dofs;
    space_.dofs_on_element(before, dofs);
    space_.dofs_on_element(after, after_dofs);

    // The rows of the cell before keep their places, and each row of the
    // cell after takes that of its degree of freedom in the cell before, or
    // a new one at the end
    const std::ptrdiff_t num_before = std::ssize(dofs);
    positions.resize(dofs.size());
    std::iota(positions.begin(), positions.end(), 0);

    for (const int dof : after_dofs) {
        const std::ptrdiff_t shared = std::distance(
            dofs.begin(), std::find(dofs.begin(), dofs.begin() + num_before,
                                    dof));

        if (shared < num_before) {
            positions.push_back(static_cast<int>(shared));
        }
        else {
            positions.push_back(static_cast<int>(dofs.size()));
            dofs.push_back(dof);
        }
    }
}

template<typename Basis, std::size_t n>
void Assembler<Basis, n>::clear()
{
    // Only the stored values, so that the pattern stays
    check_petsc(MatZeroEntries(tangent_));
    check_petsc(MatZeroEntries(mass_));
    check_petsc(VecZeroEntries(residual_));
}

template class Assembler<TensorBSpline<double, 2>, 2>;
template class Assembler<TensorBSpline<double, 3>, 3>;
template class Assembler<TensorNURBS<double, 2>, 2>;
template class Assembler<TensorNURBS<double, 3>, 3>;

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_tangent(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_tangent(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_tangent(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_tangent(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_residual(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_residual(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const PoissonElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_residual(
    const PoissonElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const PoissonElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_residual(
    const PoissonElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);

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

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_tangent(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_tangent(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_tangent(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_tangent(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_residual(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_residual(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const StokesElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_residual(
    const StokesElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const StokesElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_residual(
    const StokesElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);

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

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_tangent(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_tangent(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_tangent(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_tangent(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_residual(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_residual(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const StokesStabilizedElement<TensorBSpline<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorBSpline<double, 3>, 3>::assemble_residual(
    const StokesStabilizedElement<TensorBSpline<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const StokesStabilizedElement<TensorNURBS<double, 2>, 2>&,
    const QuadratureDomain<double, 2>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 3>, 3>::assemble_residual(
    const StokesStabilizedElement<TensorNURBS<double, 3>, 3>&,
    const QuadratureDomain<double, 3>&, const Eigen::MatrixX<double>&,
    const Eigen::VectorX<double>&, Vec);

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

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const PenaltyCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const PenaltyCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const NitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const NitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const NeumannCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const NeumannCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const NeumannCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const NeumannCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureBoundary<double, 2>&, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_tangent(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_tangent(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const Eigen::VectorX<double>&,
    const Eigen::VectorX<double>&, Vec);

template void Assembler<TensorBSpline<double, 2>, 2>::assemble_residual(
    const SkeletonNitscheCondition<TensorBSpline<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, const Eigen::VectorX<double>&, Vec);
template void Assembler<TensorNURBS<double, 2>, 2>::assemble_residual(
    const SkeletonNitscheCondition<TensorNURBS<double, 2>, 2>&,
    const QuadratureSkeleton<double, 2>&, const BoundaryProjection<double>&,
    int, const Eigen::VectorX<double>&, const Eigen::VectorX<double>&, Vec);

} // namespace iguana
