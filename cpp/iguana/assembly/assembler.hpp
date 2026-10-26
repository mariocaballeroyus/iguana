/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ASSEMBLY_ASSEMBLER_HPP
#define IGUANA_ASSEMBLY_ASSEMBLER_HPP

#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <petscmat.h>

#include "iguana/condition/condition.hpp"
#include "iguana/condition/skeleton_condition.hpp"
#include "iguana/condition/skeleton_ghost_penalty.hpp"
#include "iguana/element/element.hpp"
#include "iguana/fspace/function_space.hpp"
#include "iguana/fspace/mixed_function_space.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/quadrature/quadrature_boundary.hpp"
#include "iguana/quadrature/quadrature_domain.hpp"
#include "iguana/quadrature/quadrature_skeleton.hpp"

namespace iguana
{

/**
 * @brief Global tangent, residual and mass of a space, assembled from the
 *        local ones of elements and conditions over the points of
 *        quadratures, at coefficients of the unknowns
 *
 * The tangent and mass are sequential PETSc matrices and the residual a
 * PETSc vector, ready for its solvers after every assembly, and a failing
 * PETSc call throws std::runtime_error with its message. Each term gives
 * its residual and tangent at the coefficients a PETSc vector holds,
 * gathered for the degrees of freedom of each cell or face: a linear term
 * gives its stiffness as its tangent, and the stiffness times the
 * coefficients less its load as its residual. The sparsity pattern of the
 * tangent is fixed by the space, an entry for every pair of degrees of
 * freedom sharing an element, so it is built once and each assembly adds
 * values into it. The mass shares it, so that time integrators combine the
 * two entry by entry. With the faces coupled, it also has an entry for
 * every pair on the two cells of an interior face of the grid, as terms on
 * faces need. Assemblies add up, so that several terms, such as a domain
 * and a boundary one, build one system. The assembler does not tell linear
 * from nonlinear or transient problems, which the elements and the solvers
 * that drive it do
 *
 * The test functions of each cell may carry a weight of their own, such as
 * its volume fraction in the generalized shifted boundary method. Constant
 * on the cell, it scales the weights of its points, and a cell of weight
 * zero adds nothing and is skipped
 *
 * The bases of the space span the functions and the patch maps the
 * points, so they may differ from the patch's in degrees and continuities,
 * as long as they share the elements
 *
 * The space has one field or several, each with a basis and a block size,
 * the number of its components. On a cell its degrees of freedom run field
 * after field, and component after component within each, the layout of
 * the local matrices of the elements and conditions. With one field, the
 * PETSc matrices and vectors carry its block size, so that its solvers can
 * tell the components apart. Fields one after another share no block
 * structure, so with several the block size is one. A local matrix that
 * does not match the degrees of freedom of its cell, as one of another
 * block size, throws std::invalid_argument, since PETSc would read past
 * its end
 *
 * @tparam Basis Basis of the space and the patch, TensorBSpline or
 *         TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class Assembler
{
    static_assert(std::same_as<PetscScalar, typename Basis::Scalar>,
                  "Assembler: "
                  "PETSc must use the floating-point type of the basis");

    static_assert(std::same_as<PetscInt, int>,
                  "Assembler: "
                  "PETSc must index with the int of the degrees of freedom");

public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

    /**
     * @brief Constructs a zero tangent, residual and mass with the pattern
     *        of a space of one field
     *
     * @param space Space whose degrees of freedom number the system
     * @param patch Patch that maps the points of the quadratures
     * @param couple_faces Whether to couple the degrees of freedom of the
     *        two cells of every interior face of the grid. Every face is
     *        coupled, so that the pattern holds whichever faces a term acts
     *        on, as they change with a moving boundary
     *
     * @throws std::invalid_argument If the space does not have the elements
     *         of the patch
     * @throws std::runtime_error If PETSc fails to create the system
     *
     * @pre PETSc is initialized, and @p space and @p patch outlive the
     *      assembler, which keeps references to them
     */
    Assembler(const FunctionSpace<Basis>& space,
              const Patch<Basis, n>& patch, bool couple_faces = false);

    /**
     * @brief Constructs a zero tangent, residual and mass with the pattern
     *        of a space of several fields
     *
     * @param space Space whose degrees of freedom number the system, field
     *        after field
     * @param patch Patch that maps the points of the quadratures
     * @param couple_faces Whether to couple the degrees of freedom of the
     *        two cells of every interior face of the grid, as for a space of
     *        one field
     *
     * @throws std::invalid_argument If the space does not have the elements
     *         of the patch
     * @throws std::runtime_error If PETSc fails to create the system
     *
     * @pre PETSc is initialized, and the fields of @p space and @p patch
     *      outlive the assembler, which keeps references to them
     */
    Assembler(const MixedFunctionSpace<Basis>& space,
              const Patch<Basis, n>& patch, bool couple_faces = false);

    /// @brief Drops the references of the assembler to the tangent,
    ///        residual and mass, which PETSc destroys once nothing else
    ///        holds them
    ~Assembler();

    Assembler(const Assembler&) = delete;
    Assembler& operator=(const Assembler&) = delete;

    /**
     * @brief Adds the tangent of an element over the points of a
     *        quadrature, at coefficients
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local tangent is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of the quadrature, a
     *        row per point in its order and a column per component
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument If @p source does not have one row per
     *         point of @p quadrature, if @p coefficients does not have one
     *         entry per degree of freedom, or if the local matrices of
     *         @p element do not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis and the domain of
     *      the space, so that every element it holds has its degrees of
     *      freedom, @p source has the components @p element reads, and
     *      @p coefficients is not the residual
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_tangent(const E& element,
                          const QuadratureDomain<Scalar, dim>& quadrature,
                          const Eigen::MatrixX<Scalar>& source,
                          Vec coefficients);

    /**
     * @brief Adds the tangent of an element over the points of a
     *        quadrature, at coefficients, with the test functions of each
     *        cell weighted
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local tangent is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of the quadrature, a
     *        row per point in its order and a column per component
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the tangent of an element, or if
     *         @p cell_weights does not have one value per element of the
     *         grid
     *
     * @pre @p quadrature is built on the grid of the basis, every element it
     *      holds with a non-zero weight has its degrees of freedom in the
     *      space, and as for the tangent of an element
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_tangent(const E& element,
                          const QuadratureDomain<Scalar, dim>& quadrature,
                          const Eigen::MatrixX<Scalar>& source,
                          const Eigen::VectorX<Scalar>& cell_weights,
                          Vec coefficients);

    /**
     * @brief Adds the residual of an element over the points of a
     *        quadrature, at coefficients
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local residual is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of the quadrature, a
     *        row per point in its order and a column per component
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the tangent of an element
     *
     * @pre As for the tangent of an element
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_residual(const E& element,
                           const QuadratureDomain<Scalar, dim>& quadrature,
                           const Eigen::MatrixX<Scalar>& source,
                           Vec coefficients);

    /**
     * @brief Adds the residual of an element over the points of a
     *        quadrature, at coefficients, with the test functions of each
     *        cell weighted
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local residual is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of the quadrature, a
     *        row per point in its order and a column per component
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the weighted tangent of an
     *         element
     *
     * @pre As for the weighted tangent of an element
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_residual(const E& element,
                           const QuadratureDomain<Scalar, dim>& quadrature,
                           const Eigen::MatrixX<Scalar>& source,
                           const Eigen::VectorX<Scalar>& cell_weights,
                           Vec coefficients);

    /**
     * @brief Adds the mass of an element over the points of a quadrature
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local mass is added
     * @param quadrature Quadrature over elements of the grid of the basis
     *
     * @throws std::invalid_argument If the local matrices of @p element do
     *         not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built as for the tangent of an element
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_mass(const E& element,
                       const QuadratureDomain<Scalar, dim>& quadrature);

    /**
     * @brief Adds the mass of an element over the points of a quadrature,
     *        with the test functions of each cell weighted
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local mass is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     *
     * @throws std::invalid_argument If @p cell_weights does not have one
     *         value per element of the grid, or if the local matrices of
     *         @p element do not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built as for the weighted tangent of an element
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_mass(const E& element,
                       const QuadratureDomain<Scalar, dim>& quadrature,
                       const Eigen::VectorX<Scalar>& cell_weights);

    /**
     * @brief Adds the tangent of a condition along the points of a boundary
     *        quadrature, at coefficients
     *
     * The patch maps the weights of the quadrature, which measure the
     * boundary in parameter space, into the boundary weights the condition
     * takes, and its normals into physical ones
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local tangent is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     * @param data Data of the condition at each point of the quadrature, in
     *        its order, such as the value it imposes
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument If @p data does not have one value per
     *         point of @p quadrature, if @p coefficients does not have one
     *         entry per degree of freedom, or if the local matrices of
     *         @p condition do not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis, every element it
     *      holds has its degrees of freedom in the space, and
     *      @p coefficients is not the residual
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_tangent(const C& condition,
                          const QuadratureBoundary<Scalar, dim>& quadrature,
                          const Eigen::VectorX<Scalar>& data,
                          Vec coefficients);

    /**
     * @brief Adds the residual of a condition along the points of a
     *        boundary quadrature, at coefficients
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local residual is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     * @param data Data of the condition at each point of the quadrature, in
     *        its order, such as the value it imposes
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the tangent of a condition
     *
     * @pre As for the tangent of a condition
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_residual(const C& condition,
                           const QuadratureBoundary<Scalar, dim>& quadrature,
                           const Eigen::VectorX<Scalar>& data,
                           Vec coefficients);

    /**
     * @brief Adds the tangent of a condition along the points of a boundary
     *        quadrature, at coefficients, with the values expanded along the
     *        distance vectors of a projection
     *
     * As for a quadrature without a projection, but the condition also
     * receives the values expanded from each point to its closest point on
     * the boundary, as the shifted boundary method needs, while the weights
     * and normals stay those of the points
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local tangent is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     * @param data Data of the condition at the closest point of each point
     *        of the quadrature, in its order
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the tangent of a condition, if
     *         @p order is negative, if @p projection does not have one
     *         distance vector per point of @p quadrature, nor shift weights
     *         of one row per point and one column per multi-index when it
     *         has them, with further weights of their size alone, or if the
     *         patch is not a B-spline one
     *
     * @pre As for the tangent of a condition, and @p projection was computed
     *      from the points of @p quadrature through the patch
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_tangent(const C& condition,
                          const QuadratureBoundary<Scalar, dim>& quadrature,
                          const BoundaryProjection<Scalar>& projection,
                          int order, const Eigen::VectorX<Scalar>& data,
                          Vec coefficients) requires (dim == 2);

    /**
     * @brief Adds the residual of a condition along the points of a
     *        boundary quadrature, at coefficients, with the values expanded
     *        along the distance vectors of a projection
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local residual is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     * @param data Data of the condition at the closest point of each point
     *        of the quadrature, in its order
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the projected tangent of a
     *         condition
     *
     * @pre As for the projected tangent of a condition
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_residual(const C& condition,
                           const QuadratureBoundary<Scalar, dim>& quadrature,
                           const BoundaryProjection<Scalar>& projection,
                           int order, const Eigen::VectorX<Scalar>& data,
                           Vec coefficients) requires (dim == 2);

    /**
     * @brief Adds the tangent of a ghost penalty over the points of a
     *        skeleton quadrature, at coefficients
     *
     * On each face, the values of the cells before and after it, each with
     * its own polynomial, are differentiated across it at the same points,
     * to the degree of the field of the penalty there. The patch maps the
     * weights of the quadrature, which measure the faces in parameter
     * space, into face weights, and the local tangent couples the degrees of
     * freedom of both cells
     *
     * @param penalty Ghost penalty whose local tangent is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument If the faces are not coupled, if the map
     *         of the patch is not affine, if the patch is not a B-spline one,
     *         if the field of @p penalty is not one of the space, if a knot
     *         vector of that field repeats an interior knot, if
     *         @p coefficients does not have one entry per degree of freedom,
     *         or if the local matrices of @p penalty do not match the degrees
     *         of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis, every cell it
     *      holds has its degrees of freedom in the space, and
     *      @p coefficients is not the residual
     */
    void assemble_tangent(const SkeletonGhostPenalty<Basis, n>& penalty,
                          const QuadratureSkeleton<Scalar, dim>& quadrature,
                          Vec coefficients) requires (dim == 2);

    /**
     * @brief Adds the residual of a ghost penalty over the points of a
     *        skeleton quadrature, at coefficients
     *
     * @param penalty Ghost penalty whose local residual is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the tangent of a ghost penalty
     *
     * @pre As for the tangent of a ghost penalty
     */
    void assemble_residual(const SkeletonGhostPenalty<Basis, n>& penalty,
                           const QuadratureSkeleton<Scalar, dim>& quadrature,
                           Vec coefficients) requires (dim == 2);

    /**
     * @brief Adds the tangent of a skeleton condition over the points of a
     *        skeleton quadrature, at coefficients, with the test functions
     *        of each cell weighted
     *
     * On each face whose two cells weigh differently, the values of the
     * cells before and after it, each with its own polynomial, are filled
     * at its points. The patch maps the weights of the quadrature into face
     * weights and its normals into physical ones, pointing from the cell
     * before to the cell after. Faces whose cells weigh the same carry no
     * jump of the test functions and are skipped. Only interior faces carry
     * the condition, so the cells of non-zero weight must stay off the edge
     * of the grid, or take conditions of their own there
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local tangent is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param data Data of the condition at each point of the quadrature, in
     *        its order, such as the value it imposes
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument If the faces are not coupled, if
     *         @p data does not have one value per point of @p quadrature, if
     *         @p cell_weights does not have one value per element of the
     *         grid, if @p coefficients does not have one entry per degree of
     *         freedom, or if the local matrices of @p condition do not match
     *         the degrees of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis, every cell of a
     *      face across which the weights jump has its degrees of freedom in
     *      the space, and @p coefficients is not the residual
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_tangent(const C& condition,
                          const QuadratureSkeleton<Scalar, dim>& quadrature,
                          const Eigen::VectorX<Scalar>& data,
                          const Eigen::VectorX<Scalar>& cell_weights,
                          Vec coefficients) requires (dim == 2);

    /**
     * @brief Adds the residual of a skeleton condition over the points of a
     *        skeleton quadrature, at coefficients, with the test functions
     *        of each cell weighted
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local residual is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param data Data of the condition at each point of the quadrature, in
     *        its order, such as the value it imposes
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the tangent of a skeleton
     *         condition
     *
     * @pre As for the tangent of a skeleton condition
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_residual(const C& condition,
                           const QuadratureSkeleton<Scalar, dim>& quadrature,
                           const Eigen::VectorX<Scalar>& data,
                           const Eigen::VectorX<Scalar>& cell_weights,
                           Vec coefficients) requires (dim == 2);

    /**
     * @brief Adds the tangent of a skeleton condition over the points of a
     *        skeleton quadrature, at coefficients, with the test functions
     *        of each cell weighted and the values expanded along the
     *        distance vectors of a projection
     *
     * As for a quadrature without a projection, but the values of both
     * cells of a face are also expanded from each point to its closest
     * point on the boundary, along the same distance vectors
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local tangent is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     * @param data Data of the condition at the closest point of each point
     *        of the quadrature, in its order
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the tangent of a skeleton
     *         condition, if @p order is negative, if @p projection does not
     *         have one distance vector per point of @p quadrature, nor shift
     *         weights of one row per point and one column per multi-index
     *         when it has them, with further weights of their size alone,
     *         or if the patch is not a B-spline one
     *
     * @pre As for the tangent of a skeleton condition, and @p projection was
     *      computed from the points of @p quadrature through the patch
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_tangent(const C& condition,
                          const QuadratureSkeleton<Scalar, dim>& quadrature,
                          const BoundaryProjection<Scalar>& projection,
                          int order, const Eigen::VectorX<Scalar>& data,
                          const Eigen::VectorX<Scalar>& cell_weights,
                          Vec coefficients) requires (dim == 2);

    /**
     * @brief Adds the residual of a skeleton condition over the points of a
     *        skeleton quadrature, at coefficients, with the test functions
     *        of each cell weighted and the values expanded along the
     *        distance vectors of a projection
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local residual is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     * @param data Data of the condition at the closest point of each point
     *        of the quadrature, in its order
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     * @param coefficients Coefficients of the degrees of freedom of the
     *        space, one each
     *
     * @throws std::invalid_argument As for the projected tangent of a
     *         skeleton condition
     *
     * @pre As for the projected tangent of a skeleton condition
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_residual(const C& condition,
                           const QuadratureSkeleton<Scalar, dim>& quadrature,
                           const BoundaryProjection<Scalar>& projection,
                           int order, const Eigen::VectorX<Scalar>& data,
                           const Eigen::VectorX<Scalar>& cell_weights,
                           Vec coefficients) requires (dim == 2);

    /// @brief Zeroes the tangent, residual and mass, keeping the pattern
    void clear();

    /// @brief Tangent, of size (num_dofs, num_dofs), owned by the assembler
    ///        unless a reference to it is taken
    Mat tangent() const noexcept
    { return tangent_; }

    /// @brief Mass, of size (num_dofs, num_dofs), with the pattern of the
    ///        tangent, owned by the assembler unless a reference to it is
    ///        taken
    Mat mass() const noexcept
    { return mass_; }

    /// @brief Residual, of size num_dofs, owned by the assembler unless a
    ///        reference to it is taken
    Vec residual() const noexcept
    { return residual_; }

private:
    /**
     * @brief Adds the tangent of a condition along the points of a boundary
     *        quadrature, shifted along a projection when one is given, the
     *        work of both public overloads
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void add_condition_tangent(
        const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order,
        const Eigen::VectorX<Scalar>& data, Vec coefficients);

    /**
     * @brief Adds the residual of a condition along the points of a
     *        boundary quadrature, shifted along a projection when one is
     *        given, the work of both public overloads
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void add_condition_residual(
        const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order,
        const Eigen::VectorX<Scalar>& data, Vec coefficients);

    /**
     * @brief Adds the tangent of a skeleton condition over the points of a
     *        skeleton quadrature, shifted along a projection when one is
     *        given, the work of both public overloads
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void add_skeleton_condition_tangent(
        const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order,
        const Eigen::VectorX<Scalar>& data,
        const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
        requires (dim == 2);

    /**
     * @brief Adds the residual of a skeleton condition over the points of a
     *        skeleton quadrature, shifted along a projection when one is
     *        given, the work of both public overloads
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void add_skeleton_condition_residual(
        const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order,
        const Eigen::VectorX<Scalar>& data,
        const Eigen::VectorX<Scalar>& cell_weights, Vec coefficients)
        requires (dim == 2);

    /**
     * @brief Checks that coefficients have one entry per degree of freedom
     *        of the space
     *
     * @throws std::invalid_argument If they do not
     */
    void check_coefficients(Vec coefficients) const;

    /**
     * @brief Adds a local matrix at degrees of freedom, those of a cell or
     *        of the two cells of a face, into the tangent or the mass, the
     *        only place where they are written
     *
     * @throws std::invalid_argument If @p local does not have a row and a
     *         column per entry of @p dofs
     *
     * @pre The rows of @p local follow @p dofs, and @p matrix is the
     *      tangent or the mass
     */
    void add_to_matrix(Mat matrix, std::span<const int> dofs,
                       const Eigen::MatrixX<Scalar>& local);

    /**
     * @brief Adds a local residual at degrees of freedom, those of a cell or
     *        of the two cells of a face, the only place where the residual
     *        is written
     *
     * @throws std::invalid_argument If @p local does not have an entry per
     *         entry of @p dofs
     *
     * @pre The rows of @p local follow @p dofs
     */
    void add_to_residual(std::span<const int> dofs,
                         const Eigen::VectorX<Scalar>& local);

    /**
     * @brief Finishes the writes into the tangent or the mass, so that PETSc
     *        can use it, at the end of every public assembly into it
     */
    void finalize(Mat matrix);

    /**
     * @brief Finishes the writes into the residual, so that PETSc can use
     *        it, at the end of every public assembly into it
     */
    void finalize(Vec vector);

    /**
     * @brief Degrees of freedom of the two cells of a face, each once, and
     *        the one of each row of the terms on the face
     *
     * The rows of a term on a face are those of the cell before it followed
     * by those of the cell after it, so that a function active on both cells
     * has two rows, which share its degree of freedom
     *
     * @param before Cell before the face
     * @param after Cell after the face
     * @param dofs Output, overwritten, those of the cell before followed by
     *        those of the cell after that the cell before does not have
     * @param positions Output, overwritten, the index in @p dofs of the
     *        degree of freedom of each row
     */
    void face_dofs(int before, int after, std::vector<int>& dofs,
                   std::vector<int>& positions) const;

    /// @brief Space whose degrees of freedom number the system, a view of
    ///        its fields
    MixedFunctionSpace<Basis> space_;

    /// @brief Patch that maps the points of the quadratures
    const Patch<Basis, n>& patch_;

    /// @brief Whether the pattern couples the cells of every interior face
    bool couple_faces_;

    /// @brief Tangent, with an entry for every pair of degrees of freedom
    ///        sharing an element, or the cells of a face when coupled
    Mat tangent_ = nullptr;

    /// @brief Mass, with the pattern of the tangent
    Mat mass_ = nullptr;

    /// @brief Residual
    Vec residual_ = nullptr;
};

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ASSEMBLER_HPP
