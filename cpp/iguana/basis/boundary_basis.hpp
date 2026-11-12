/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_BASIS_BOUNDARY_BASIS_HPP
#define IGUANA_BASIS_BOUNDARY_BASIS_HPP

#include <concepts>
#include <vector>

#include <Eigen/Core>

#include "iguana/basis/bspline.hpp"

namespace iguana
{

/**
 * @brief Basis of a field on the faces of a boundary, a B-spline basis in
 *        the parameter of each face, its functions numbered face after face
 *
 * @details The functions of a face vanish on the others, so that the field
 *          may jump where two faces meet. In the parameters of the faces,
 *          the basis is the same wherever the boundary lies, whatever grid
 *          it crosses. On a face that is a closed curve, the basis may wrap
 *          around: its B-spline extends p spans past each end of the
 *          interval, with spans that repeat after one period, and each of
 *          its last p functions is the first p ones again, so that the field
 *          is smooth across the point where the curve closes
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
class BoundaryBasis
{
public:
    /**
     * @brief Constructs the basis from that of each face
     *
     * @param bases Basis of each face, on the interval of its parameter
     * @param periodic Whether the basis of each face wraps around, none
     *        when empty
     *
     * @throws std::invalid_argument If @p periodic is neither empty nor of
     *         one flag per face, if a periodic basis has no more than 2p
     *         functions, keeping no more than p once wrapped, or if the
     *         spans of its knots do not repeat after one period
     */
    explicit BoundaryBasis(std::vector<BSpline<T>> bases,
                           std::vector<bool> periodic = {});

    /// @brief Number of functions, of all the faces
    constexpr int num_functions() const noexcept
    { return offsets_.back(); }

    /// @brief Number of faces
    constexpr int num_faces() const noexcept
    { return static_cast<int>(bases_.size()); }

    /**
     * @brief First function of a face, those of the face running up to the
     *        first of the next one
     *
     * @param face Face, num_faces() giving num_functions()
     *
     * @pre @p face lies in [0, num_faces()]
     */
    constexpr int first_function(int face) const noexcept
    { return offsets_[static_cast<std::size_t>(face)]; }

    /**
     * @brief Integral of each function over the parameter of its face
     *
     * @details A function of a periodic face that wraps around gathers the
     *          parts of two functions of its B-spline inside the interval,
     *          which add up to the whole of the second by periodicity
     *
     * @return One integral per function
     */
    Eigen::VectorX<T> integrals() const;

    /**
     * @brief Checks that points lie on faces of the basis, inside the
     *        interval of the basis of their face up to round-off
     *
     * @param faces Face of each point
     * @param parameters Parameter of each point on its face
     *
     * @throws std::invalid_argument If a point lies on a face without a
     *         basis, or outside the interval of its basis
     */
    void check(const Eigen::VectorXi& faces,
               const Eigen::VectorX<T>& parameters) const;

    /**
     * @brief Functions active at a point of a face, and their values there
     *
     * @param face Face of the point
     * @param parameter Parameter of the point on its face
     * @param functions Output, overwritten
     * @param values Output, the value of each, overwritten
     *
     * @pre The point passes check()
     */
    void active_at(int face, T parameter, std::vector<int>& functions,
                   Eigen::VectorX<T>& values) const;

    /**
     * @brief Functions active at a range of points, each once in increasing
     *        order
     *
     * @param faces Face of each point
     * @param parameters Parameter of each point on its face
     * @param first First point of the range
     * @param count Number of points of the range
     * @param functions Output, overwritten
     *
     * @pre The points of the range pass check()
     */
    void active_on(const Eigen::VectorXi& faces,
                   const Eigen::VectorX<T>& parameters, int first, int count,
                   std::vector<int>& functions) const;

    /**
     * @brief Values at points of the field of some coefficients
     *
     * @param coefficients Coefficient of each function
     * @param faces Face of each point
     * @param parameters Parameter of each point on its face
     * @return One value per point
     *
     * @pre @p coefficients has one entry per function, and the points pass
     *      check()
     */
    Eigen::VectorX<T> evaluate(const Eigen::VectorX<T>& coefficients,
                               const Eigen::VectorXi& faces,
                               const Eigen::VectorX<T>& parameters) const;

private:
    /// @brief Basis of each face
    std::vector<BSpline<T>> bases_;

    /// @brief Whether the basis of each face wraps around
    std::vector<bool> periodic_;

    /// @brief First function of each face, followed by their number
    std::vector<int> offsets_;
};

} // namespace iguana

#endif // IGUANA_BASIS_BOUNDARY_BASIS_HPP
