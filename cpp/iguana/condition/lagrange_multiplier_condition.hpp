/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_CONDITION_LAGRANGE_MULTIPLIER_CONDITION_HPP
#define IGUANA_CONDITION_LAGRANGE_MULTIPLIER_CONDITION_HPP

#include <cstddef>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "iguana/assembly/element_values.hpp"
#include "iguana/basis/boundary_basis.hpp"
#include "iguana/basis/bspline.hpp"
#include "iguana/element/trace.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/quadrature/quadrature_boundary.hpp"

namespace iguana
{

/**
 * @brief Condition imposing values on a trace through Lagrange multipliers,
 *        unknowns of its own along a boundary
 *
 *     ( λ, T(v) )_Γ + ( μ, T(u) − g )_Γ
 *
 * @details The multipliers are B-splines on each face of the boundary, in
 *          the parameter of the face, numbered face after face and with no
 *          continuity across corners. They must be coarser than the cells,
 *          for the saddle point to be stable. With the components of a
 *          vector trace, the condition imposes its normal component T(u)·n
 *          with one multiplier instead. Its unknowns belong to no element,
 *          so that it is not a Condition: on the points of each cell, it
 *          gives the coupling of the multipliers active there with the
 *          degrees of freedom of the cell, and the load of the multipliers,
 *          which the assembler adds with the transpose of the coupling. The
 *          multipliers follow the boundary as it moves, as they live on its
 *          faces rather than on the grid
 *
 * @tparam Basis Basis of the patch, TensorBSpline or TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class LagrangeMultiplierCondition final
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

    /**
     * @brief Constructs the condition imposing values on a trace
     *
     * @param trace Trace whose values are imposed
     * @param bases Basis of the multipliers on each face of the boundary, on
     *        the interval of its parameter
     * @param quadrature Quadrature of the boundary the condition acts along
     *
     * @throws std::invalid_argument If a point of the quadrature lies on a
     *         face without a basis, or outside the interval of its basis
     *
     * @pre @p trace and @p quadrature outlive the condition, which keeps
     *      references to them
     */
    LagrangeMultiplierCondition(
        const Trace<Basis, n>& trace, std::vector<BSpline<Scalar>> bases,
        const QuadratureBoundary<Scalar, dim>& quadrature);

    /**
     * @brief Constructs the condition imposing the normal component of a
     *        vector trace
     *
     * @param components Traces of the n components, in order
     * @param bases Basis of the multipliers on each face of the boundary, on
     *        the interval of its parameter
     * @param quadrature Quadrature of the boundary the condition acts along
     *
     * @throws std::invalid_argument If there are not n components, or as
     *         for a trace
     *
     * @pre @p components and @p quadrature outlive the condition, which
     *      keeps references to them
     */
    LagrangeMultiplierCondition(
        std::vector<std::reference_wrapper<const Trace<Basis, n>>> components,
        std::vector<BSpline<Scalar>> bases,
        const QuadratureBoundary<Scalar, dim>& quadrature);

    /// @brief Number of multipliers, of all the faces
    constexpr int num_multipliers() const noexcept
    { return basis_.num_functions(); }

    /// @brief Whether the condition imposes the normal component of a vector
    ///        trace
    constexpr bool normal() const noexcept
    { return traces_.size() > 1; }

    /// @brief Quadrature of the boundary the condition acts along
    constexpr const QuadratureBoundary<Scalar, dim>& quadrature() const
        noexcept
    { return quadrature_; }

    /// @brief Optional values the traces read from ElementValues
    ValueFlags flags() const noexcept;

    /**
     * @brief Multipliers active at points of the quadrature, each once in
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
     * @brief Coupling of the multipliers active at the points of a cell with
     *        the degrees of freedom of the cell
     *
     * @param values Values at the points
     * @param weights Boundary weights, one per point
     * @param normals Unit normals of the boundary in physical space, of size
     *        (num_points, n)
     * @param first First of the points in the quadrature
     * @param coupling Output with a row per multiplier of active_on() and a
     *        column per degree of freedom of the cell, ordered as for
     *        Element, overwritten
     *
     * @pre @p values were filled with flags() at the points of the
     *      quadrature from @p first, as many as @p weights
     */
    void local_coupling(const ElementValues<Basis, n>& values,
                        const Eigen::VectorX<Scalar>& weights,
                        const PointMatrix<Scalar, n>& normals, int first,
                        Eigen::MatrixX<Scalar>& coupling) const;

    /**
     * @brief Load of the multipliers active at the points of a cell, the
     *        imposed values tested by each
     *
     * @param weights Boundary weights, one per point
     * @param data Value imposed at each point, the normal component for a
     *        vector trace
     * @param first First of the points in the quadrature
     * @param load Output with an entry per multiplier of active_on(),
     *        overwritten
     *
     * @pre @p weights and @p data hold the points of the quadrature from
     *      @p first
     */
    void local_load(const Eigen::VectorX<Scalar>& weights,
                    const Eigen::VectorX<Scalar>& data, int first,
                    Eigen::VectorX<Scalar>& load) const;

    /**
     * @brief Value of the multipliers at each point of the quadrature
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
    /// @brief Trace whose values are imposed, or the n components whose
    ///        normal component is
    std::vector<std::reference_wrapper<const Trace<Basis, n>>> traces_;

    /// @brief Basis of the multipliers on the faces of the boundary
    BoundaryBasis<Scalar> basis_;

    /// @brief Quadrature of the boundary the condition acts along
    const QuadratureBoundary<Scalar, dim>& quadrature_;
};

} // namespace iguana

#endif // IGUANA_CONDITION_LAGRANGE_MULTIPLIER_CONDITION_HPP
