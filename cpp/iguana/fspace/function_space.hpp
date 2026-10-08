/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_FSPACE_FUNCTION_SPACE_HPP
#define IGUANA_FSPACE_FUNCTION_SPACE_HPP

#include <cstddef>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/embedding/cell_classification.hpp"
#include "iguana/fspace/dof_map.hpp"

namespace iguana
{

/**
 * @brief Standard space of a basis on a physical domain, with one degree of
 *        freedom per function active on it
 *
 * A function active on a cell that is not outside the physical domain gets
 * a degree of freedom, numbered in increasing function index, and one
 * supported on outside cells alone gets none. On a cell that is not
 * outside, the k-th degree of freedom belongs to the k-th function of the
 * active_on_element() of the basis, so that the basis values pair with them
 * directly. Outside cells list no degrees of freedom
 *
 * The space covers the cells its classification does not mark outside, so
 * a method integrating a smaller region, such as the inside cells alone,
 * passes the classification of that region
 *
 * @tparam Basis Basis whose functions span the space, TensorBSpline or
 *         TensorNURBS
 */
template<typename Basis>
class FunctionSpace
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = Basis::dimension;

    /**
     * @brief Constructs the space of a basis on a physical domain embedded
     *        in its elements
     *
     * @param basis Basis whose functions span the space
     * @param classification Cell type of each element of the basis grid.
     *        With every cell inside, each function gets the degree of
     *        freedom of its own index
     *
     * @throws std::invalid_argument If the classification does not have one
     *         cell type per element
     */
    FunctionSpace(Basis basis,
                  const CellClassification<Scalar, dimension>& classification);

    /// @brief Basis whose functions span the space
    constexpr const Basis& basis() const noexcept
    { return basis_; }

    /// @brief Degrees of freedom of each element of the basis grid
    constexpr const DofMap& dof_map() const noexcept
    { return dof_map_; }

private:
    /// @brief Basis whose functions span the space
    Basis basis_;

    /// @brief Degrees of freedom of each element, built from the basis
    DofMap dof_map_;
};

} // namespace iguana

#endif // IGUANA_FSPACE_FUNCTION_SPACE_HPP
