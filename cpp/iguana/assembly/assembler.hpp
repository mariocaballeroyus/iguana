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
 * @brief Global stiffness, mass and load of a space, assembled from the
 *        local ones of elements over the points of quadratures
 *
 * The stiffness and mass are sequential PETSc matrices and the load a PETSc
 * vector, ready for its solvers after every assembly, and a failing PETSc
 * call throws std::runtime_error with its message. The sparsity pattern
 * of the stiffness is fixed by the space, an entry for every pair of
 * degrees of freedom sharing an element, so it is built once and each
 * assembly adds values into it. The mass shares it, so that time
 * integrators combine the two entry by entry. With the faces coupled, it
 * also has an entry for every pair on the two cells of an interior face of
 * the grid, as terms on faces need. Assemblies add up, so that
 * several terms, such as a domain and a boundary one, build one system.
 * The assembler does not tell linear from nonlinear or transient problems,
 * which the elements and the solvers that drive it do
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
     * @brief Constructs a zero stiffness and load with the pattern of a space
     *        of one field
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
     * @brief Constructs a zero stiffness and load with the pattern of a space
     *        of several fields
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

    /// @brief Drops the references of the assembler to the stiffness, mass
    ///        and load, which PETSc destroys once nothing else holds them
    ~Assembler();

    Assembler(const Assembler&) = delete;
    Assembler& operator=(const Assembler&) = delete;

    /**
     * @brief Adds the stiffness of an element over the points of a
     *        quadrature
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local stiffness is added
     * @param quadrature Quadrature over elements of the grid of the basis
     *
     * @throws std::invalid_argument If the local matrices of @p element do
     *         not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis and the domain of
     *      the space, so that every element it holds has its degrees of
     *      freedom
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_stiffness(const E& element,
                            const QuadratureDomain<Scalar, dim>& quadrature);

    /**
     * @brief Adds the stiffness of an element over the points of a
     *        quadrature, with the test functions of each cell weighted
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local stiffness is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     *
     * @throws std::invalid_argument If @p cell_weights does not have one
     *         value per element of the grid, or if the local matrices of
     *         @p element do not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis, and every
     *      element it holds with a non-zero weight has its degrees of
     *      freedom in the space
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_stiffness(const E& element,
                            const QuadratureDomain<Scalar, dim>& quadrature,
                            const Eigen::VectorX<Scalar>& cell_weights);

    /**
     * @brief Adds the load of an element over the points of a quadrature,
     *        from a source given at them
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local load is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of the quadrature, a
     *        row per point in its order and a column per component
     *
     * @throws std::invalid_argument If @p source does not have one row per
     *         point of @p quadrature, or if the local loads of @p element do
     *         not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built as for assemble_stiffness(), and
     *      @p source has the components @p element reads
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_load(const E& element,
                       const QuadratureDomain<Scalar, dim>& quadrature,
                       const Eigen::MatrixX<Scalar>& source);

    /**
     * @brief Adds the load of an element over the points of a quadrature,
     *        from a source given at them, with the test functions of each
     *        cell weighted
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local load is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of the quadrature, a
     *        row per point in its order and a column per component
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     *
     * @throws std::invalid_argument If @p source does not have one row per
     *         point of @p quadrature, if @p cell_weights does not have one
     *         value per element of the grid, or if the local loads of
     *         @p element do not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built as for the weighted assemble_stiffness(),
     *      and @p source has the components @p element reads
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_load(const E& element,
                       const QuadratureDomain<Scalar, dim>& quadrature,
                       const Eigen::MatrixX<Scalar>& source,
                       const Eigen::VectorX<Scalar>& cell_weights);

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
     * @pre @p quadrature is built as for assemble_stiffness()
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
     * @pre @p quadrature is built as for the weighted assemble_stiffness()
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_mass(const E& element,
                       const QuadratureDomain<Scalar, dim>& quadrature,
                       const Eigen::VectorX<Scalar>& cell_weights);

    /**
     * @brief Adds the stiffness of a condition along the points of a
     *        boundary quadrature
     *
     * The patch maps the weights of the quadrature, which measure the
     * boundary in parameter space, into the boundary weights the condition
     * takes, and its normals into physical ones
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local stiffness is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     *
     * @throws std::invalid_argument If the local matrices of @p condition do
     *         not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis, and every element
     *      it holds has its degrees of freedom in the space
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_stiffness(const C& condition,
                            const QuadratureBoundary<Scalar, dim>& quadrature);

    /**
     * @brief Adds the load of a condition along the points of a boundary
     *        quadrature, from data given at them
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local load is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     * @param data Value the condition imposes at each point of the
     *        quadrature, in its order
     *
     * @throws std::invalid_argument If @p data does not have one value per
     *         point of @p quadrature, or if the local loads of @p condition
     *         do not match the degrees of freedom of the space
     *
     * @pre @p quadrature is built as for the stiffness of a condition
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_load(const C& condition,
                       const QuadratureBoundary<Scalar, dim>& quadrature,
                       const Eigen::VectorX<Scalar>& data);

    /**
     * @brief Adds the stiffness of a condition along the points of a
     *        boundary quadrature, with the values expanded along the
     *        distance vectors of a projection
     *
     * As for a quadrature without a projection, but the condition also
     * receives the values expanded from each point to its closest point on
     * the boundary, as the shifted boundary method needs, while the weights
     * and normals stay those of the points
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local stiffness is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     *
     * @throws std::invalid_argument If @p order is negative, if
     *         @p projection does not have one distance vector per point of
     *         @p quadrature, nor shift weights of one row per point and one
     *         column per multi-index when it has them, if the patch is not a
     *         B-spline one, or if the local matrices of @p condition do not
     *         match the degrees of freedom of the space
     *
     * @pre @p quadrature is built as for the stiffness of a condition, and
     *      @p projection was computed from its points through the patch
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_stiffness(const C& condition,
                            const QuadratureBoundary<Scalar, dim>& quadrature,
                            const BoundaryProjection<Scalar>& projection,
                            int order) requires (dim == 2);

    /**
     * @brief Adds the load of a condition along the points of a boundary
     *        quadrature, from data given at their closest points, with the
     *        values expanded along the distance vectors of a projection
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local load is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     * @param data Value the condition imposes at the closest point of each
     *        point of the quadrature, in its order
     *
     * @throws std::invalid_argument If @p data does not have one value per
     *         point of @p quadrature, or as for the projected stiffness of a
     *         condition
     *
     * @pre As for the projected stiffness of a condition
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_load(const C& condition,
                       const QuadratureBoundary<Scalar, dim>& quadrature,
                       const BoundaryProjection<Scalar>& projection,
                       int order, const Eigen::VectorX<Scalar>& data)
        requires (dim == 2);

    /**
     * @brief Adds the stiffness of a ghost penalty over the points of a
     *        skeleton quadrature
     *
     * On each face, the values of the cells before and after it, each with
     * its own polynomial, are differentiated across it at the same points,
     * to the degree of the field of the penalty there. The patch maps the
     * weights of the quadrature, which measure the faces in parameter
     * space, into face weights, and the local stiffness couples the degrees
     * of freedom of both cells
     *
     * @param penalty Ghost penalty whose local stiffness is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     *
     * @throws std::invalid_argument If the faces are not coupled, if the map
     *         of the patch is not affine, if the patch is not a B-spline one,
     *         if the field of @p penalty is not one of the space, if a knot
     *         vector of that field repeats an interior knot, or if the local
     *         matrices of @p penalty do not match the degrees of freedom of
     *         the space
     *
     * @pre @p quadrature is built on the grid of the basis, and every cell
     *      it holds has its degrees of freedom in the space
     */
    void assemble_stiffness(const SkeletonGhostPenalty<Basis, n>& penalty,
                            const QuadratureSkeleton<Scalar, dim>& quadrature)
        requires (dim == 2);

    /**
     * @brief Adds the stiffness of a skeleton condition over the points of a
     *        skeleton quadrature, with the test functions of each cell
     *        weighted
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
     * @param condition Skeleton condition whose local stiffness is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     *
     * @throws std::invalid_argument If the faces are not coupled, if
     *         @p cell_weights does not have one value per element of the
     *         grid, or if the local matrices of @p condition do not match
     *         the degrees of freedom of the space
     *
     * @pre @p quadrature is built on the grid of the basis, and every cell
     *      of a face across which the weights jump has its degrees of
     *      freedom in the space
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_stiffness(const C& condition,
                            const QuadratureSkeleton<Scalar, dim>& quadrature,
                            const Eigen::VectorX<Scalar>& cell_weights)
        requires (dim == 2);

    /**
     * @brief Adds the load of a skeleton condition over the points of a
     *        skeleton quadrature, from data given at them, with the test
     *        functions of each cell weighted
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local load is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param data Value the condition imposes at each point of the
     *        quadrature, in its order
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     *
     * @throws std::invalid_argument If @p data does not have one value per
     *         point of @p quadrature, or as for the stiffness of a skeleton
     *         condition
     *
     * @pre @p quadrature is built as for the stiffness of a skeleton condition
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_load(const C& condition,
                       const QuadratureSkeleton<Scalar, dim>& quadrature,
                       const Eigen::VectorX<Scalar>& data,
                       const Eigen::VectorX<Scalar>& cell_weights)
        requires (dim == 2);

    /**
     * @brief Adds the stiffness of a skeleton condition over the points of a
     *        skeleton quadrature, with the test functions of each cell
     *        weighted and the values expanded along the distance vectors of
     *        a projection
     *
     * As for a quadrature without a projection, but the values of both
     * cells of a face are also expanded from each point to its closest
     * point on the boundary, along the same distance vectors
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local stiffness is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     *
     * @throws std::invalid_argument As for the stiffness of a skeleton
     *         condition, if @p order is negative, if @p projection does not
     *         have one distance vector per point of @p quadrature, nor shift
     *         weights of one row per point and one column per multi-index
     *         when it has them, or if the patch is not a B-spline one
     *
     * @pre As for the stiffness of a skeleton condition, and @p projection was
     *      computed from the points of @p quadrature through the patch
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_stiffness(const C& condition,
                            const QuadratureSkeleton<Scalar, dim>& quadrature,
                            const BoundaryProjection<Scalar>& projection,
                            int order,
                            const Eigen::VectorX<Scalar>& cell_weights)
        requires (dim == 2);

    /**
     * @brief Adds the load of a skeleton condition over the points of a
     *        skeleton quadrature, from data given at their closest points,
     *        with the test functions of each cell weighted and the values
     *        expanded along the distance vectors of a projection
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local load is added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary the condition is shifted from
     * @param order Highest total order of the Taylor series, or of the
     *        derivatives that the shift weights of @p projection combine, as
     *        in ElementValues::shift()
     * @param data Value the condition imposes at the closest point of each
     *        point of the quadrature, in its order
     * @param cell_weights Weight of the test functions on each element of
     *        the grid, in its numbering
     *
     * @throws std::invalid_argument If @p data does not have one value per
     *         point of @p quadrature, or as for the projected stiffness of a
     *         skeleton condition
     *
     * @pre As for the projected stiffness of a skeleton condition
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void assemble_load(const C& condition,
                       const QuadratureSkeleton<Scalar, dim>& quadrature,
                       const BoundaryProjection<Scalar>& projection,
                       int order, const Eigen::VectorX<Scalar>& data,
                       const Eigen::VectorX<Scalar>& cell_weights)
        requires (dim == 2);

    /// @brief Zeroes the stiffness, mass and load, keeping the pattern
    void clear();

    /// @brief Stiffness, of size (num_dofs, num_dofs), owned by the
    ///        assembler unless a reference to it is taken
    Mat stiffness() const noexcept
    { return stiffness_; }

    /// @brief Mass, of size (num_dofs, num_dofs), with the pattern of the
    ///        stiffness, owned by the assembler unless a reference to it is
    ///        taken
    Mat mass() const noexcept
    { return mass_; }

    /// @brief Load, of size num_dofs, owned by the assembler unless a
    ///        reference to it is taken
    Vec load() const noexcept
    { return load_; }

private:
    /**
     * @brief Adds the stiffness of a condition along the points of a
     *        boundary quadrature, shifted along a projection when one is
     *        given, the work of both public overloads
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void add_condition_stiffness(
        const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order);

    /**
     * @brief Adds the load of a condition along the points of a boundary
     *        quadrature, shifted along a projection when one is given, the
     *        work of both public overloads
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void add_condition_load(
        const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order,
        const Eigen::VectorX<Scalar>& data);

    /**
     * @brief Adds the stiffness of a skeleton condition over the points of a
     *        skeleton quadrature, shifted along a projection when one is
     *        given, the work of both public overloads
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void add_skeleton_condition_stiffness(
        const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order,
        const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2);

    /**
     * @brief Adds the load of a skeleton condition over the points of a
     *        skeleton quadrature, shifted along a projection when one is
     *        given, the work of both public overloads
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void add_skeleton_condition_load(
        const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>* projection, int order,
        const Eigen::VectorX<Scalar>& data,
        const Eigen::VectorX<Scalar>& cell_weights) requires (dim == 2);

    /**
     * @brief Adds a local matrix at degrees of freedom, those of a cell or
     *        of the two cells of a face, into the stiffness or the mass, the
     *        only place where they are written
     *
     * @throws std::invalid_argument If @p local does not have a row and a
     *         column per entry of @p dofs
     *
     * @pre The rows of @p local follow @p dofs, and @p matrix is the
     *      stiffness or the mass
     */
    void add_to_matrix(Mat matrix, std::span<const int> dofs,
                       const Eigen::MatrixX<Scalar>& local);

    /**
     * @brief Adds a local load at degrees of freedom, those of a cell or of
     *        the two cells of a face, the only place where the load is
     *        written
     *
     * @throws std::invalid_argument If @p local does not have an entry per
     *         entry of @p dofs
     *
     * @pre The rows of @p local follow @p dofs
     */
    void add_to_load(std::span<const int> dofs,
                     const Eigen::VectorX<Scalar>& local);

    /**
     * @brief Finishes the writes into the stiffness, mass and load, so that
     *        PETSc can use them, at the end of every public assembly
     */
    void finalize();

    /**
     * @brief Degrees of freedom of the two cells of a face, those of the
     *        cell before it followed by those of the cell after it, the order
     *        of the rows of the terms on faces
     *
     * @param before Cell before the face
     * @param after Cell after the face
     * @param dofs Output, overwritten
     */
    void face_dofs(int before, int after, std::vector<int>& dofs) const;

    /// @brief Space whose degrees of freedom number the system, a view of
    ///        its fields
    MixedFunctionSpace<Basis> space_;

    /// @brief Patch that maps the points of the quadratures
    const Patch<Basis, n>& patch_;

    /// @brief Whether the pattern couples the cells of every interior face
    bool couple_faces_;

    /// @brief Stiffness, with an entry for every pair of degrees of freedom
    ///        sharing an element, or the cells of a face when coupled
    Mat stiffness_ = nullptr;

    /// @brief Mass, with the pattern of the stiffness
    Mat mass_ = nullptr;

    /// @brief Load
    Vec load_ = nullptr;
};

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ASSEMBLER_HPP
