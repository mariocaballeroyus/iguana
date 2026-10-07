/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ASSEMBLY_ASSEMBLER_HPP
#define IGUANA_ASSEMBLY_ASSEMBLER_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "iguana/condition/condition.hpp"
#include "iguana/element/element.hpp"
#include "iguana/fspace/function_space.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/quadrature/boundary_quadrature.hpp"
#include "iguana/quadrature/domain_quadrature.hpp"

namespace iguana
{

/**
 * @brief Global stiffness and load of a space, assembled from the local
 *        ones of elements over the points of quadratures
 *
 * The sparsity pattern of the stiffness is fixed by the space, an entry
 * for every pair of degrees of freedom sharing an element, so it is built
 * once and each assembly adds values into it. Assemblies add up, so that
 * several terms, such as a domain and a boundary one, build one system.
 * The assembler does not tell linear from nonlinear or transient problems,
 * which the elements and the solvers that drive it do
 *
 * The basis of the space spans the functions and that of the patch maps the
 * points, so both must be one basis, as in isoparametric analysis
 *
 * @tparam Basis Basis of the space and the patch, TensorBSpline or
 *         TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class Assembler
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

    /**
     * @brief Constructs a zero stiffness and load with the pattern of a space
     *
     * @param space Space whose degrees of freedom number the system
     * @param patch Patch that maps the points of the quadratures
     *
     * @pre @p space and @p patch share one basis, and both outlive the
     *      assembler, which keeps references to them
     */
    Assembler(const FunctionSpace<Basis>& space,
              const Patch<Basis, n>& patch);

    /**
     * @brief Adds the stiffness of an element over the points of a
     *        quadrature
     *
     * @tparam E Element, a final class derived from Element
     *
     * @param element Element whose local stiffness is added
     * @param quadrature Quadrature over elements of the grid of the basis
     *
     * @pre @p quadrature is built on the grid of the basis and the domain of
     *      the space, so that every element it holds has its degrees of
     *      freedom
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_stiffness(const E& element,
                            const DomainQuadrature<Scalar, dim>& quadrature);

    /**
     * @brief Adds the load of an element over the points of a quadrature,
     *        from a source given at them
     *
     * @tparam E Element, a final class derived from Element
     *
     * @param element Element whose local load is added
     * @param quadrature Quadrature over elements of the grid of the basis
     * @param source Value of the source at each point of the quadrature, in
     *        its order
     *
     * @throws std::invalid_argument If @p source does not have one value per
     *         point of @p quadrature
     *
     * @pre @p quadrature is built as for assemble_stiffness()
     */
    template<std::derived_from<Element<Basis, n>> E>
    void assemble_load(const E& element,
                       const DomainQuadrature<Scalar, dim>& quadrature,
                       const Eigen::VectorX<Scalar>& source);

    /**
     * @brief Adds the stiffness of a condition along the points of a
     *        boundary quadrature
     *
     * The patch maps the weights of the quadrature, which measure the
     * boundary in parameter space, into the boundary weights the condition
     * takes, through the normals of the boundary. On a shifted quadrature
     * the values are expanded towards the boundary it is shifted onto, as
     * the shifted boundary method needs, while the weights stay those of
     * its points
     *
     * @tparam C Condition, a final class derived from Condition
     *
     * @param condition Condition whose local stiffness is added
     * @param quadrature Quadrature of a boundary embedded in the grid of the
     *        basis
     *
     * @throws std::invalid_argument If @p quadrature is shifted and the patch
     *         is not a B-spline one
     *
     * @pre @p quadrature is built on the grid of the basis, and every element
     *      it holds has its degrees of freedom in the space
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_stiffness(const C& condition,
                            const BoundaryQuadrature<Scalar, dim>& quadrature);

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
     *        quadrature, in its order, given at its closest point on a
     *        shifted quadrature
     *
     * @throws std::invalid_argument If @p data does not have one value per
     *         point of @p quadrature, or if @p quadrature is shifted and the
     *         patch is not a B-spline one
     *
     * @pre @p quadrature is built as for the stiffness of a condition
     */
    template<std::derived_from<Condition<Basis, n>> C>
    void assemble_load(const C& condition,
                       const BoundaryQuadrature<Scalar, dim>& quadrature,
                       const Eigen::VectorX<Scalar>& data);

    /// @brief Zeroes the stiffness and load, keeping the pattern
    void clear();

    /// @brief Stiffness, of size (num_dofs, num_dofs)
    constexpr const Eigen::SparseMatrix<Scalar, Eigen::RowMajor>&
    stiffness() const noexcept
    { return stiffness_; }

    /// @brief Load, of size num_dofs
    constexpr const Eigen::VectorX<Scalar>& load() const noexcept
    { return load_; }

private:
    /**
     * @brief Adds a local stiffness at the degrees of freedom of a cell, the
     *        only place where the stiffness is written
     *
     * @pre The rows of @p local follow the degrees of freedom of the cell
     */
    void add_to_stiffness(int cell, const Eigen::MatrixX<Scalar>& local);

    /**
     * @brief Adds a local load at the degrees of freedom of a cell, the only
     *        place where the load is written
     *
     * @pre The rows of @p local follow the degrees of freedom of the cell
     */
    void add_to_load(int cell, const Eigen::VectorX<Scalar>& local);

    /// @brief Space whose degrees of freedom number the system
    const FunctionSpace<Basis>& space_;

    /// @brief Patch that maps the points of the quadratures
    const Patch<Basis, n>& patch_;

    /// @brief Stiffness, with an entry for every pair of degrees of freedom
    ///        sharing an element
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> stiffness_;

    /// @brief Load
    Eigen::VectorX<Scalar> load_;
};

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ASSEMBLER_HPP
