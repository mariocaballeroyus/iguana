/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ASSEMBLY_FORM_HPP
#define IGUANA_ASSEMBLY_FORM_HPP

#include <concepts>
#include <cstddef>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "iguana/assembly/assembler.hpp"
#include "iguana/condition/condition.hpp"
#include "iguana/condition/skeleton_condition.hpp"
#include "iguana/condition/skeleton_ghost_penalty.hpp"
#include "iguana/element/element.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/quadrature/quadrature_boundary.hpp"
#include "iguana/quadrature/quadrature_domain.hpp"
#include "iguana/quadrature/quadrature_skeleton.hpp"

namespace iguana
{

/**
 * @brief Integrals of a problem over the cells, boundaries and faces of a
 *        grid, whose tangents, residuals and masses an assembler adds
 *
 * @details Domain integrals run over the cells of a QuadratureDomain,
 *          boundary integrals along a QuadratureBoundary and skeleton
 *          integrals across the faces of a QuadratureSkeleton. Each keeps
 *          references to what it integrates and views of its data, so that
 *          the data may change between assemblies without building the form
 *          again. The tangents, residuals and masses are separate passes,
 *          as a nonlinear solver evaluates the residual more often than the
 *          tangent
 *
 * @tparam Basis Basis of the space and the patch, TensorBSpline or
 *         TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class Form
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

    /// @brief Data an integral reads at every assembly, one value per point
    ///        or per cell, whose values may change in between
    using Data = Eigen::Ref<const Eigen::VectorX<Scalar>>;

    /// @brief Source an element reads at every assembly, a row per point
    ///        and a column per component, whose values may change in
    ///        between
    using Source = Eigen::Ref<const Eigen::MatrixX<Scalar>>;

    /**
     * @brief Adds the tangent and residual of an element over a domain
     *        quadrature, with the test functions of each cell weighted
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local tangent and residual are added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of @p quadrature, a
     *        row per point and a column per component
     * @param cell_weights Weight of the test functions on each element of
     *        the grid
     *
     * @pre What the integral reads outlives the form, and @p source and
     *      @p cell_weights keep their sizes. The assembly checks them as
     *      Assembler::assemble_residual() does
     */
    template<std::derived_from<Element<Basis, n>> E>
    void add_domain_integral(const E& element,
                             const QuadratureDomain<Scalar, dim>& quadrature,
                             Source source, Data cell_weights)
    {
        const SourceView sources = view(source);
        const Eigen::Map<const Eigen::VectorX<Scalar>> weights =
            view(cell_weights);

        tangents_.push_back([&element, &quadrature, sources, weights](
                                Assembler<Basis, n>& assembler,
                                Vec coefficients) {
            assembler.assemble_tangent(element, quadrature, sources, weights,
                                       coefficients);
        });
        residuals_.push_back([&element, &quadrature, sources, weights](
                                 Assembler<Basis, n>& assembler,
                                 Vec coefficients) {
            assembler.assemble_residual(element, quadrature, sources, weights,
                                        coefficients);
        });
    }

    /**
     * @brief Adds the mass of an element over a domain quadrature, with the
     *        test functions of each cell weighted
     *
     * @tparam E Element, a class derived from Element
     *
     * @param element Element whose local mass is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param cell_weights Weight of the test functions on each element of
     *        the grid
     *
     * @pre As for add_domain_integral()
     */
    template<std::derived_from<Element<Basis, n>> E>
    void add_mass_integral(const E& element,
                           const QuadratureDomain<Scalar, dim>& quadrature,
                           Data cell_weights)
    {
        const Eigen::Map<const Eigen::VectorX<Scalar>> weights =
            view(cell_weights);

        masses_.push_back([&element, &quadrature, weights](
                              Assembler<Basis, n>& assembler) {
            assembler.assemble_mass(element, quadrature, weights);
        });
    }

    /**
     * @brief Adds the tangent and residual of a condition along a boundary
     *        quadrature
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local tangent and residual are added
     * @param quadrature Quadrature of a boundary in the grid of the basis
     * @param data Data of the condition at each point of @p quadrature
     *
     * @pre What the integral reads outlives the form, and @p data keeps its
     *      size. The assembly checks it as Assembler::assemble_residual()
     *      does
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void add_boundary_integral(
        const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
        Data data)
    {
        const Eigen::Map<const Eigen::VectorX<Scalar>> values = view(data);

        tangents_.push_back([&condition, &quadrature, values](
                                Assembler<Basis, n>& assembler,
                                Vec coefficients) {
            assembler.assemble_tangent(condition, quadrature, values,
                                       coefficients);
        });
        residuals_.push_back([&condition, &quadrature, values](
                                 Assembler<Basis, n>& assembler,
                                 Vec coefficients) {
            assembler.assemble_residual(condition, quadrature, values,
                                        coefficients);
        });
    }

    /**
     * @brief Adds the tangent and residual of a condition along a boundary
     *        quadrature, with the values expanded along the distance vectors
     *        of a projection
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local tangent and residual are added
     * @param quadrature Quadrature of a boundary in the grid of the basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary where the data is given
     * @param order Total order of the Taylor expansion, or of the
     *        derivatives that the shift weights of @p projection combine
     * @param data Data of the condition at the closest point of each point
     *        of @p quadrature
     *
     * @pre As for add_boundary_integral(). The assembly checks the
     *      projection and order as Assembler::assemble_residual() does
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void add_boundary_integral(
        const C& condition, const QuadratureBoundary<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>& projection, int order, Data data)
        requires (dim == 2)
    {
        const Eigen::Map<const Eigen::VectorX<Scalar>> values = view(data);

        tangents_.push_back([&condition, &quadrature, &projection, order,
                             values](Assembler<Basis, n>& assembler,
                                     Vec coefficients) {
            assembler.assemble_tangent(condition, quadrature, projection,
                                       order, values, coefficients);
        });
        residuals_.push_back([&condition, &quadrature, &projection, order,
                              values](Assembler<Basis, n>& assembler,
                                      Vec coefficients) {
            assembler.assemble_residual(condition, quadrature, projection,
                                        order, values, coefficients);
        });
    }

    /**
     * @brief Adds the tangent and residual of a skeleton condition across
     *        the faces of a skeleton quadrature, with the test functions of
     *        each cell weighted and the values expanded along the distance
     *        vectors of a projection
     *
     * @tparam C Skeleton condition, a final class derived from
     *         SkeletonCondition
     *
     * @param condition Skeleton condition whose local tangent and residual
     *        are added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary where the data is given
     * @param order Total order of the Taylor expansion, or of the
     *        derivatives that the shift weights of @p projection combine
     * @param data Value the condition imposes at the closest point of each
     *        point of @p quadrature
     * @param cell_weights Weight of the test functions on each element of
     *        the grid
     *
     * @pre What the integral reads outlives the form, and @p data and
     *      @p cell_weights keep their sizes. The assembly checks them as
     *      Assembler::assemble_residual() does
     */
    template<std::derived_from<SkeletonCondition<Basis, n>> C>
    void add_skeleton_integral(
        const C& condition, const QuadratureSkeleton<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>& projection, int order, Data data,
        Data cell_weights) requires (dim == 2)
    {
        const Eigen::Map<const Eigen::VectorX<Scalar>> values = view(data);
        const Eigen::Map<const Eigen::VectorX<Scalar>> weights =
            view(cell_weights);

        tangents_.push_back([&condition, &quadrature, &projection, order,
                             values, weights](Assembler<Basis, n>& assembler,
                                              Vec coefficients) {
            assembler.assemble_tangent(condition, quadrature, projection,
                                       order, values, weights, coefficients);
        });
        residuals_.push_back([&condition, &quadrature, &projection, order,
                              values, weights](Assembler<Basis, n>& assembler,
                                               Vec coefficients) {
            assembler.assemble_residual(condition, quadrature, projection,
                                        order, values, weights, coefficients);
        });
    }

    /**
     * @brief Adds the tangent and residual of a ghost penalty across the
     *        faces of a skeleton quadrature
     *
     * @param penalty Ghost penalty whose local tangent and residual are
     *        added
     * @param quadrature Quadrature over interior faces of the grid of the
     *        basis
     *
     * @pre @p penalty and @p quadrature outlive the form. The assembly checks
     *      them as Assembler::assemble_tangent() does
     */
    void add_skeleton_integral(
        const SkeletonGhostPenalty<Basis, n>& penalty,
        const QuadratureSkeleton<Scalar, dim>& quadrature) requires (dim == 2)
    {
        tangents_.push_back([&penalty, &quadrature](
                                Assembler<Basis, n>& assembler,
                                Vec coefficients) {
            assembler.assemble_tangent(penalty, quadrature, coefficients);
        });
        residuals_.push_back([&penalty, &quadrature](
                                 Assembler<Basis, n>& assembler,
                                 Vec coefficients) {
            assembler.assemble_residual(penalty, quadrature, coefficients);
        });
    }

    /**
     * @brief Adds the tangent of every integral of the form into an
     *        assembler, at coefficients and with the data as it is now
     *
     * @param assembler Assembler of the space and patch of the integrals,
     *        added into as each of its assemblies is
     * @param coefficients Coefficients of the unknowns, one per degree of
     *        freedom of the space
     *
     * @throws std::invalid_argument As the assemblies of the integrals do
     */
    void assemble_tangent(Assembler<Basis, n>& assembler,
                          Vec coefficients) const
    {
        for (const Integral& integral : tangents_)
            integral(assembler, coefficients);
    }

    /**
     * @brief Adds the residual of every integral of the form into an
     *        assembler, at coefficients and with the data as it is now
     *
     * @param assembler Assembler of the space and patch of the integrals,
     *        added into as each of its assemblies is
     * @param coefficients Coefficients of the unknowns, one per degree of
     *        freedom of the space
     *
     * @throws std::invalid_argument As the assemblies of the integrals do
     */
    void assemble_residual(Assembler<Basis, n>& assembler,
                           Vec coefficients) const
    {
        for (const Integral& integral : residuals_)
            integral(assembler, coefficients);
    }

    /**
     * @brief Adds the mass of every mass integral of the form into an
     *        assembler
     *
     * @param assembler Assembler of the space and patch of the integrals,
     *        added into as each of its assemblies is
     *
     * @throws std::invalid_argument As the assemblies of the integrals do
     */
    void assemble_mass(Assembler<Basis, n>& assembler) const
    {
        for (const MassIntegral& integral : masses_)
            integral(assembler);
    }

private:
    /// @brief Assembly of one integral into an assembler, at coefficients
    using Integral = std::function<void(Assembler<Basis, n>&, Vec)>;

    /// @brief Assembly of the mass of one integral into an assembler
    using MassIntegral = std::function<void(Assembler<Basis, n>&)>;

    /// @brief View of a source, its columns as far apart as in the source
    using SourceView =
        Eigen::Map<const Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>>;

    /// @brief View of data, read where it lies at every assembly
    static Eigen::Map<const Eigen::VectorX<Scalar>> view(Data data)
    { return {data.data(), data.size()}; }

    /// @brief View of a source, read where it lies at every assembly
    static SourceView view(Source source)
    {
        return {source.data(), source.rows(), source.cols(),
                Eigen::OuterStride<>(source.outerStride())};
    }

    /// @brief Tangents of the integrals, in the order they were added
    std::vector<Integral> tangents_;

    /// @brief Residuals of the integrals, in the order they were added
    std::vector<Integral> residuals_;

    /// @brief Masses of the mass integrals, in the order they were added
    std::vector<MassIntegral> masses_;
};

} // namespace iguana

#endif // IGUANA_ASSEMBLY_FORM_HPP
