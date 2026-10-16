/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_FSPACE_FUNCTION_SPACE_HPP
#define IGUANA_FSPACE_FUNCTION_SPACE_HPP

#include <cstddef>
#include <vector>

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
 * A vector field has a block size, its number of components, and each
 * function carries one degree of freedom per component. Component c of
 * the k-th degree of freedom of the map is bs k + c, so the components of a
 * function are adjacent, while on an element they run component after
 * component, the layout of the local matrices
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
     * @param block_size Number of components of the field
     *
     * @throws std::invalid_argument If the classification does not have one
     *         cell type per element, or if the block size is not positive
     */
    FunctionSpace(Basis basis,
                  const CellClassification<Scalar, dimension>& classification,
                  int block_size = 1);

    /// @brief Basis whose functions span the space
    constexpr const Basis& basis() const noexcept
    { return basis_; }

    /// @brief Degrees of freedom of each element of the basis grid, one per
    ///        function whatever the block size
    constexpr const DofMap& dof_map() const noexcept
    { return dof_map_; }

    /// @brief Number of components of the field
    constexpr int block_size() const noexcept
    { return block_size_; }

    /// @brief Number of degrees of freedom, those of the map times the
    ///        block size
    constexpr int num_dofs() const noexcept
    { return block_size_ * dof_map_.num_dofs(); }

    /**
     * @brief Degrees of freedom of the functions active on an element, those
     *        of their first component, then of their second, and so on
     *
     * @param element Element index, in the numbering of the grid
     * @param dofs Output of size block_size() times the number of active
     *        functions, overwritten
     *
     * @pre @p element lies in [0, dof_map().num_elements())
     */
    void dofs_on_element(int element, std::vector<int>& dofs) const;

private:
    /// @brief Basis whose functions span the space
    Basis basis_;

    /// @brief Degrees of freedom of each element, built from the basis
    DofMap dof_map_;

    /// @brief Number of components of the field
    int block_size_;
};

} // namespace iguana

#endif // IGUANA_FSPACE_FUNCTION_SPACE_HPP
