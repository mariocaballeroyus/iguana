/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_FSPACE_MIXED_FUNCTION_SPACE_HPP
#define IGUANA_FSPACE_MIXED_FUNCTION_SPACE_HPP

#include <cstddef>
#include <functional>
#include <vector>

#include "iguana/fspace/function_space.hpp"

namespace iguana
{

/**
 * @brief Space of several fields on the elements of one grid, each a
 *        FunctionSpace with a basis of its own
 *
 * The fields may differ in degrees, continuities, classifications and
 * block sizes, as long as their bases share the elements. Their degrees of
 * freedom run field after field, each field taking a contiguous range from
 * its offset(), and on an element they also run field after field, each as
 * its space lists them, the layout of the local matrices. Split into
 * parts, the degrees of freedom would run part after part, and field after
 * field within each
 *
 * The space views its fields: it keeps references to them, and its copies
 * view the same fields
 *
 * @tparam Basis Basis of every field, TensorBSpline or TensorNURBS
 */
template<typename Basis>
class MixedFunctionSpace
{
public:
    /// @brief Floating-point type of the bases
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = Basis::dimension;

    /**
     * @brief Constructs the space of several fields
     *
     * @param fields Spaces of the fields, in the order of their degrees of
     *        freedom
     *
     * @throws std::invalid_argument If there is no field, or if a field
     *         does not have the elements of the first
     *
     * @pre The fields outlive the space, which keeps references to them
     */
    explicit MixedFunctionSpace(
        std::vector<std::reference_wrapper<const FunctionSpace<Basis>>> fields);

    /// @brief Number of fields
    constexpr std::size_t num_fields() const noexcept
    { return fields_.size(); }

    /**
     * @brief Space of a field
     *
     * @param field Field index, in the order of the degrees of freedom
     *
     * @pre @p field lies in [0, num_fields())
     */
    constexpr const FunctionSpace<Basis>& field(std::size_t field) const
        noexcept
    { return fields_[field]; }

    /**
     * @brief First degree of freedom of a field
     *
     * @param field Field index, in the order of the degrees of freedom
     *
     * @pre @p field lies in [0, num_fields())
     */
    constexpr int offset(std::size_t field) const noexcept
    { return offsets_[field]; }

    /// @brief Number of degrees of freedom of all the fields
    constexpr int num_dofs() const noexcept
    { return offsets_.back(); }

    /// @brief Number of elements, in the numbering of the grid
    constexpr int num_elements() const noexcept
    { return fields_.front().get().dof_map().num_elements(); }

    /// @brief Bases of the fields, in their order
    std::vector<std::reference_wrapper<const Basis>> bases() const;

    /**
     * @brief Degrees of freedom of the functions active on an element, those
     *        of the first field, then of the second, and so on, each as its
     *        space lists them
     *
     * @param element Element index, in the numbering of the grid
     * @param dofs Output, overwritten
     *
     * @pre @p element lies in [0, num_elements())
     */
    void dofs_on_element(int element, std::vector<int>& dofs) const;

private:
    /// @brief Spaces of the fields, in the order of their degrees of freedom
    std::vector<std::reference_wrapper<const FunctionSpace<Basis>>> fields_;

    /// @brief First degree of freedom of each field, followed by the number
    ///        of degrees of freedom
    std::vector<int> offsets_;
};

} // namespace iguana

#endif // IGUANA_FSPACE_MIXED_FUNCTION_SPACE_HPP
