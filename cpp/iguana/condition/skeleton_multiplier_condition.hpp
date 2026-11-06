/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_SKELETON_MULTIPLIER_CONDITION_HPP
#define IGUANA_CONDITION_SKELETON_MULTIPLIER_CONDITION_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/basis/boundary_basis.hpp"
#include "iguana/basis/bspline.hpp"
#include "iguana/element/flux.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/quadrature/quadrature_skeleton.hpp"

namespace iguana
{

/**
 * @brief Condition imposing the velocity of a fluid across the faces where
 *        the weights of the test functions jump, its normal component
 *        through Lagrange multipliers on the true boundary and its
 *        tangential one by Nitsche's method
 *
 *     − j ( v, {{2μ ε(u) n}}_γ )_F
 *     + s j ( {{2μ ε(v) n}}_γ · τ, ({{S u}}_γ − g) · τ )_F
 *     + β |j| ( {{S v}}_γ · τ, ({{S u}}_γ − g) · τ / h )_F
 *     + j ( v · n, λ ∘ s − {{S p}}_γ + {{p}}_γ )_F
 *     + j ( μ ∘ s, ({{S u}}_γ − g) · n )_F
 *
 * @details The counterpart of LagrangeMultiplierCondition in the
 *          Generalized Shifted Boundary Method, whose weighted volume terms
 *          leave their boundary terms on the faces where the weights jump:
 *          the multipliers are integrated there, each point of a face
 *          reading them at the parameter of its closest point on the
 *          boundary, so that they stay B-splines on the faces of the
 *          boundary, whatever cells it cuts as it moves. They stand for the
 *          pressure of the flux of the faces, carried from the boundary to
 *          each point by the Taylor series of the pressure, and impose the
 *          normal component of the shifted velocity. The tangential
 *          component, along the tangent τ of the boundary at the closest
 *          point, takes Nitsche's penalty and adjoint term, and nothing is
 *          tested by the pressure functions, so that a divergence-conforming
 *          velocity stays free of divergence. The multipliers must be
 *          coarser than the cells, for the saddle point to be stable
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class SkeletonMultiplierCondition final
{
    // The tangent of the boundary is the normal turned a quarter
    static_assert(n == 2, "SkeletonMultiplierCondition: "
                          "the boundary must be a curve in the plane");

public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

    /// @brief Traces of the components of the velocity, in order
    using Traces = std::vector<std::reference_wrapper<const Trace<Basis, n>>>;

    /// @brief Fluxes of the components of the velocity, in order
    using Fluxes = std::vector<std::reference_wrapper<const Flux<Basis, n>>>;

    /**
     * @brief Constructs the condition imposing the velocity of a fluid
     *
     * @param velocity Traces of the n components of the velocity
     * @param fluxes Viscous fluxes of the n components, the traction
     *        without the pressure
     * @param pressure Trace of the pressure
     * @param bases Basis of the multipliers on each face of the boundary,
     *        on the interval of its parameter
     * @param quadrature Quadrature over the faces where the weights may
     *        jump
     * @param projection Projection of the points of @p quadrature onto the
     *        boundary
     * @param order Highest total order of the Taylor series to the closest
     *        points
     * @param penalty Penalty β of the tangential velocity, which the size of
     *        the cells scales, positive for the symmetric method and
     *        non-negative for the skew-symmetric one
     * @param symmetric Whether the adjoint term is that of the symmetric
     *        method, rather than the skew-symmetric one
     *
     * @throws std::invalid_argument If there are not n traces and n fluxes,
     *         if the order is negative, if the penalty is negative, or zero
     *         for the symmetric method, if a closest point lies on a face
     *         without a basis or outside the interval of its basis, or if no
     *         closest point reads a multiplier
     *
     * @pre The traces, the fluxes, @p quadrature and @p projection outlive
     *      the condition, which keeps references to them
     */
    SkeletonMultiplierCondition(
        Traces velocity, Fluxes fluxes, const Trace<Basis, n>& pressure,
        std::vector<BSpline<Scalar>> bases,
        const QuadratureSkeleton<Scalar, dim>& quadrature,
        const BoundaryProjection<Scalar>& projection, int order,
        Scalar penalty, bool symmetric = true);

    /// @brief Number of multipliers, of all the faces of the boundary
    constexpr int num_multipliers() const noexcept
    { return basis_.num_functions(); }

    /// @brief Quadrature over the faces where the weights may jump
    constexpr const QuadratureSkeleton<Scalar, dim>& quadrature() const
        noexcept
    { return quadrature_; }

    /// @brief Projection of the points of the quadrature onto the boundary
    constexpr const BoundaryProjection<Scalar>& projection() const noexcept
    { return projection_; }

    /// @brief Highest total order of the Taylor series
    constexpr int order() const noexcept
    { return order_; }

    /// @brief Penalty β of the tangential velocity
    constexpr Scalar penalty() const noexcept
    { return penalty_; }

    /// @brief Whether the adjoint term is that of the symmetric method
    constexpr bool symmetric() const noexcept
    { return symmetric_; }

    /// @brief Optional values the traces and fluxes read from ElementValues
    ValueFlags flags() const noexcept;

    /**
     * @brief Multipliers read at points of the quadrature, each once in
     *        increasing order
     *
     * @param first First point
     * @param count Number of points
     * @param multipliers Output, overwritten
     *
     * @pre The points lie in the quadrature
     */
    void active_on(int first, int count, std::vector<int>& multipliers) const;

    /**
     * @brief Tangent of the velocity terms on one face, a row and a column
     *        per degree of freedom of the cell before it and then of the cell
     *        after it, as for SkeletonCondition
     *
     * @param before Values of the cell before the face at its points
     * @param after Values of the cell after the face at the same points
     * @param shifted_before Values of the cell before the face expanded to
     *        the closest points
     * @param shifted_after Values of the cell after the face expanded to the
     *        closest points
     * @param cell_weights Weights of the test functions on the cell before
     *        the face and on the cell after it
     * @param weights Face weights, one per point
     * @param normals Unit normals of the face in physical space, from the
     *        cell before it to the cell after it, of size (num_points, n)
     * @param first First of the points in the quadrature
     * @param tangent Output, overwritten
     *
     * @pre The values were filled with flags() at the points of the
     *      quadrature from @p first, as many as @p weights, and the cell
     *      weights are not both zero
     */
    void local_tangent(const ElementValues<Basis, n>& before,
                       const ElementValues<Basis, n>& after,
                       const ElementValues<Basis, n>& shifted_before,
                       const ElementValues<Basis, n>& shifted_after,
                       const std::array<Scalar, 2>& cell_weights,
                       const Eigen::VectorX<Scalar>& weights,
                       const PointMatrix<Scalar, n>& normals, int first,
                       Eigen::MatrixX<Scalar>& tangent) const;

    /**
     * @brief Load of the velocity terms on one face, the tangential velocity
     *        imposed, on the rows of local_tangent()
     *
     * @param before Values of the cell before the face at its points
     * @param after Values of the cell after the face at the same points
     * @param shifted_before Values of the cell before the face expanded to
     *        the closest points
     * @param shifted_after Values of the cell after the face expanded to the
     *        closest points
     * @param cell_weights Weights of the test functions on the two cells
     * @param weights Face weights, one per point
     * @param normals Unit normals of the face in physical space, as for
     *        local_tangent()
     * @param data Velocity imposed at the closest point of each point, of
     *        size (num_points, n)
     * @param first First of the points in the quadrature
     * @param load Output, overwritten
     *
     * @pre As for local_tangent()
     */
    void local_load(const ElementValues<Basis, n>& before,
                    const ElementValues<Basis, n>& after,
                    const ElementValues<Basis, n>& shifted_before,
                    const ElementValues<Basis, n>& shifted_after,
                    const std::array<Scalar, 2>& cell_weights,
                    const Eigen::VectorX<Scalar>& weights,
                    const PointMatrix<Scalar, n>& normals,
                    const Eigen::MatrixX<Scalar>& data, int first,
                    Eigen::VectorX<Scalar>& load) const;

    /**
     * @brief Coupling of the multipliers read on one face with the test
     *        functions of its two cells, a row per multiplier of active_on()
     *        and a column per row of local_tangent()
     *
     * @param before Values of the cell before the face at its points
     * @param after Values of the cell after the face at the same points
     * @param cell_weights Weights of the test functions on the two cells
     * @param weights Face weights, one per point
     * @param normals Unit normals of the face in physical space, as for
     *        local_tangent()
     * @param first First of the points in the quadrature
     * @param coupling Output, overwritten
     *
     * @pre As for local_tangent()
     */
    void local_coupling(const ElementValues<Basis, n>& before,
                        const ElementValues<Basis, n>& after,
                        const std::array<Scalar, 2>& cell_weights,
                        const Eigen::VectorX<Scalar>& weights,
                        const PointMatrix<Scalar, n>& normals, int first,
                        Eigen::MatrixX<Scalar>& coupling) const;

    /**
     * @brief Constraint of the multipliers read on one face on the normal
     *        component of the shifted velocity, a row per multiplier of
     *        active_on() and a column per row of local_tangent(), and its
     *        load, the normal component of the velocity imposed
     *
     * @param shifted_before Values of the cell before the face expanded to
     *        the closest points
     * @param shifted_after Values of the cell after the face expanded to the
     *        closest points
     * @param cell_weights Weights of the test functions on the two cells
     * @param weights Face weights, one per point
     * @param normals Unit normals of the face in physical space, as for
     *        local_tangent()
     * @param data Velocity imposed at the closest point of each point, of
     *        size (num_points, n)
     * @param first First of the points in the quadrature
     * @param constraint Output, overwritten
     * @param load Output, an entry per row of @p constraint, overwritten
     *
     * @pre As for local_tangent()
     */
    void local_constraint(const ElementValues<Basis, n>& shifted_before,
                          const ElementValues<Basis, n>& shifted_after,
                          const std::array<Scalar, 2>& cell_weights,
                          const Eigen::VectorX<Scalar>& weights,
                          const PointMatrix<Scalar, n>& normals,
                          const Eigen::MatrixX<Scalar>& data, int first,
                          Eigen::MatrixX<Scalar>& constraint,
                          Eigen::VectorX<Scalar>& load) const;

    /**
     * @brief Value of the multipliers at the closest point of each point of
     *        the quadrature
     *
     * @param multipliers Coefficient of each multiplier
     * @return One value per point
     *
     * @throws std::invalid_argument If there is not one coefficient per
     *         multiplier
     */
    Eigen::VectorX<Scalar> values(
        const Eigen::VectorX<Scalar>& multipliers) const;

private:
    /**
     * @brief Values at the points of a face of the multipliers read there,
     *        a row per multiplier of active_on() and a column per point
     */
    Eigen::MatrixX<Scalar> multiplier_values(int first, int count) const;

    /// @brief Traces of the components of the velocity
    Traces velocity_;

    /// @brief Viscous fluxes of the components of the velocity
    Fluxes fluxes_;

    /// @brief Trace of the pressure
    const Trace<Basis, n>& pressure_;

    /// @brief Basis of the multipliers on the faces of the boundary
    BoundaryBasis<Scalar> basis_;

    /// @brief Quadrature over the faces where the weights may jump
    const QuadratureSkeleton<Scalar, dim>& quadrature_;

    /// @brief Projection of its points onto the boundary
    const BoundaryProjection<Scalar>& projection_;

    /// @brief Highest total order of the Taylor series
    int order_;

    /// @brief Penalty β of the tangential velocity
    Scalar penalty_;

    /// @brief Whether the adjoint term is that of the symmetric method
    bool symmetric_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_SKELETON_MULTIPLIER_CONDITION_HPP
