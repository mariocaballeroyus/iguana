/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP
#define IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP

#include<array>
#include<concepts>

#include<Eigen/Core>

namespace iguana
{

/**
 * @brief Tensor-product Gauss-Legendre rule of a parametric dimension.
 *
 * The rule holds its points on the reference box \f$ [-1, 1]^d \f$ and
 * maps them onto element boxes on demand.
 * 
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the points and weights.
 */
template<int d, std::floating_point T>
class GaussLegendre
{
    static_assert(d >= 1, "GaussLegendre: "
                          "the parametric dimension must be positive");

public:
    /// @brief The number of parametric directions, \f$ d \f$.
    static constexpr int dimension = d;

    /// @brief The largest tabulated number of points per direction.
    static constexpr int max_points = 8;

public:
    /**
     * @brief Constructs the rule with the same count in every direction.
     *
     * @param num_points Points per direction, in [1, #max_points].
     *
     * @throws std::invalid_argument If the count is out of the table.
     */
    explicit GaussLegendre(int num_points);

    /**
     * @brief Constructs the rule with one count per direction.
     *
     * A count of \f$ n \f$ integrates polynomials of the direction
     * exactly up to degree \f$ 2n - 1 \f$.
     *
     * @param num_points Points of each direction, in [1, #max_points].
     *
     * @throws std::invalid_argument If a count is out of the table.
     */
    explicit GaussLegendre(const std::array<int, d>& num_points);

    /// @brief The number of points, over all the directions.
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief Reference points, of size (num_points,dimension).
    constexpr const Eigen::MatrixX<T>& points() const noexcept
    { return points_; }

    /// @brief Reference weights, of size (num_points,).
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

    /**
     * @brief Maps the rule onto a given element box.
     *
     * Every direction maps affinely from \f$ [-1, 1] \f$ onto its side
     * of the box, and the weights are scaled by the volume ratio.
     *
     * @param start The parameters at which the element starts, as given
     *        by element_start().
     * @param end The parameters at which the element ends, as given by
     *        element_end().
     * @param points Output buffer of size (num_points,dimension), as the
     *        basis evaluation consumes it. Resized if its shape changes.
     * @param weights Output buffer of size (num_points,), resized if its
     *        size changes.
     *
     * @pre The box is not empty in any direction. It is not checked.
     */
    void map_to(const std::array<T, d>& start, const std::array<T, d>& end,
                Eigen::MatrixX<T>& points, Eigen::VectorX<T>& weights)
        const;

private:
    /// @brief Reference points, of size (num_points,dimension).
    Eigen::MatrixX<T> points_;

    /// @brief Reference weights, of size (num_points,).
    Eigen::VectorX<T> weights_;
};

extern template class GaussLegendre<1, double>;
extern template class GaussLegendre<2, double>;
extern template class GaussLegendre<3, double>;

} // namespace iguana

#endif // IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP
