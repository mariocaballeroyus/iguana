/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_ASSEMBLY_ELEMENT_DATA_HPP
#define IGUANA_ASSEMBLY_ELEMENT_DATA_HPP

#include<array>
#include<concepts>
#include<vector>

#include "iguana/domain/tensor_domain.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/quadrature/gauss_legendre.hpp"

namespace iguana
{

/**
 * @brief The quantities an ElementData fills on reinit(), one flag per
 *        quantity.
 *
 * Every integrand of a problem declares the quantities its weak form
 * reads, and the assembler combines them, so that nothing is computed
 * that nobody reads.
 */
enum UpdateFlags : unsigned
{
    /// @brief Nothing beyond the points and the actives.
    update_none = 0,

    /// @brief Values of the active functions.
    update_values = 1u << 0,

    /// @brief First derivatives of the active functions.
    update_derivatives = 1u << 1,

    /// @brief Tangents of the map, its covariant basis.
    update_tangents = 1u << 2,

    /// @brief The metric at the points, the first fundamental form.
    update_metric = 1u << 3,

    /// @brief The contravariant metric at the points.
    update_metric_inverse = 1u << 4,

    /// @brief The measures at the points, \f$ \sqrt{\det g} \f$.
    update_measure = 1u << 5,

    /// @brief Quadrature weights scaled by the measure.
    update_weights = 1u << 6,

    /// @brief Physical positions of the points.
    update_positions = 1u << 7,
};

/// @brief Combines two sets of flags.
constexpr UpdateFlags operator|(UpdateFlags a, UpdateFlags b) noexcept
{
    return static_cast<UpdateFlags>(static_cast<unsigned>(a)
                                    | static_cast<unsigned>(b));
}

/// @brief The flags a set holds of another, testable as a boolean.
constexpr UpdateFlags operator&(UpdateFlags a, UpdateFlags b) noexcept
{
    return static_cast<UpdateFlags>(static_cast<unsigned>(a)
                                    & static_cast<unsigned>(b));
}

/**
 * @brief Expands a request down the quantity dependency ladder.
 *
 * @param flags Requested quantities.
 *
 * @return The request plus every quantity it consumes.
 */
constexpr UpdateFlags expand_flags(UpdateFlags flags) noexcept
{
    if (flags & update_weights)
        flags = flags | update_measure;
    if (flags & (update_measure | update_metric_inverse))
        flags = flags | update_metric;
    if (flags & update_metric)
        flags = flags | update_tangents;
    if (flags & update_tangents)
        flags = flags | update_derivatives;
    if (flags & update_positions)
        flags = flags | update_values;
    return flags;
}

/**
 * @brief The highest derivative order an expanded request consumes.
 *
 * @param flags Expanded quantity flags.
 *
 * @return The order the basis is evaluated to, or -1 for none.
 */
constexpr int required_order(UpdateFlags flags) noexcept
{
    if (flags & (update_derivatives | update_tangents))
        return 1;
    if (flags & update_values)
        return 0;
    return -1;
}

/**
 * @brief The data of one domain element, prepared for the integrands.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the fields.
 */
template<int d, std::floating_point T>
class ElementData
{
public:
    /**
     * @brief Constructs the data of a problem.
     *
     * @param domain The domain whose elements are visited.
     * @param patch The geometry of the domain. Its basis is the basis
     *        of the unknown, the isoparametric choice.
     * @param rule The quadrature rule applied on every element.
     * @param flags The quantities reinit() fills, combined over the
     *        integrands of the problem and expanded here.
     */
    ElementData(const TensorDomain<d, T>& domain,
                const Patch<d, T>& patch,
                const GaussLegendre<d, T>& rule,
                UpdateFlags flags) noexcept
        : domain_(&domain), patch_(&patch), rule_(&rule),
          flags_(expand_flags(flags)), order_(required_order(flags_))
    {
    }

    /**
     * @brief Refills the data for one element.
     *
     * @param element The element handle of a walk over the domain.
     */
    void reinit(const TensorDomainIterator<d, T>& element);

    /// @brief The quantities, refilled in place by reinit().
    Eigen::VectorXi actives;
    Eigen::MatrixX<T> points;
    Eigen::VectorX<T> weights;
    std::vector<Eigen::MatrixX<T>> values;
    std::array<Eigen::MatrixX3<T>, d> tangents;
    Eigen::MatrixX<T> metric;
    Eigen::MatrixX<T> metric_inverse;
    Eigen::VectorX<T> measures;
    Eigen::MatrixX3<T> positions;

private:
    /// @brief The domain whose elements are visited.
    const TensorDomain<d, T>* domain_;

    /// @brief The geometry of the domain.
    const Patch<d, T>* patch_;

    /// @brief The quadrature rule applied on every element.
    const GaussLegendre<d, T>* rule_;

    /// @brief The expanded quantities reinit() fills.
    UpdateFlags flags_;

    /// @brief The order the basis is evaluated to, -1 for none.
    int order_;

};

extern template class ElementData<1, double>;
extern template class ElementData<2, double>;
extern template class ElementData<3, double>;

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ELEMENT_DATA_HPP
