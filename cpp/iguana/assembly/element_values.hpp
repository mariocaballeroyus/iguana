/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ASSEMBLY_ELEMENT_VALUES_HPP
#define IGUANA_ASSEMBLY_ELEMENT_VALUES_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "iguana/basis/tensor_bspline.hpp"
#include "iguana/geometry/boundary_projection.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Optional values of ElementValues, computed only when flagged
 *
 * The values, parametric gradients, tangents and measures are always
 * computed, as every integral needs the measure and the measure needs the
 * rest
 */
struct ValueFlags
{
    /// @brief Gradients in physical space, J^-T times the parametric ones,
    ///        which only a domain patch has
    bool physical_gradients = false;

    /// @brief Second derivatives in physical space, which only a domain patch
    ///        has, along with the physical gradients they need
    bool physical_hessians = false;
};

/**
 * @brief Values of the functions of one or several bases and of the map of
 *        a patch at points of one element at a time
 *
 * The functions are those of bases on the elements of the patch, which
 * maps the points, one basis for each field of a space. The map is
 * evaluated once for all of them, and when a basis is the patch's own, as
 * in isoparametric analysis, one evaluation of it gives both its functions
 * and the Jacobian J of the map. reinit() moves the values to an element
 * and fills them there, reusing the buffers of the previous element. The
 * rows of a field follow active_on_element() of its basis, so that the
 * k-th row pairs with the k-th degree of freedom a space of that basis
 * lists on the element
 *
 * @tparam Basis Basis of the patch and of the functions, TensorBSpline or
 *         TensorNURBS
 * @tparam n Dimension of the physical space of the patch
 */
template<typename Basis, std::size_t n>
class ElementValues
{
public:
    /// @brief Floating-point type of the basis
    using Scalar = typename Basis::Scalar;

    /// @brief Number of parametric directions
    static constexpr std::size_t dim = Basis::dimension;

    /**
     * @brief Constructs the values of the functions of a patch, filled by
     *        reinit()
     *
     * @param patch Patch whose functions are evaluated and which maps the
     *        points
     * @param flags Optional values to compute
     *
     * @throws std::invalid_argument If physical gradients or Hessians are
     *         flagged on a patch with fewer directions than its space, such
     *         as a shell
     *
     * @pre @p patch outlives the values, which keep a reference to it
     */
    ElementValues(const Patch<Basis, n>& patch, ValueFlags flags);

    /**
     * @brief Constructs the values of the functions of a basis on the
     *        elements of a patch, filled by reinit()
     *
     * @param patch Patch that maps the points
     * @param basis Basis whose functions are evaluated, of any degrees and
     *        continuities on the elements of the patch
     * @param flags Optional values to compute
     *
     * @throws std::invalid_argument If the basis does not have the elements
     *         of the patch, or if physical gradients or Hessians are flagged
     *         on a patch with fewer directions than its space
     *
     * @pre @p patch and @p basis outlive the values, which keep references
     *      to them
     */
    ElementValues(const Patch<Basis, n>& patch, const Basis& basis,
                  ValueFlags flags);

    /**
     * @brief Constructs the values of the functions of several bases on the
     *        elements of a patch, filled by reinit()
     *
     * @param patch Patch that maps the points
     * @param bases Bases whose functions are evaluated, one per field, each
     *        of any degrees and continuities on the elements of the patch
     * @param flags Optional values to compute, for every basis
     *
     * @throws std::invalid_argument If there is no basis, if a basis does
     *         not have the elements of the patch, or if physical gradients
     *         or Hessians are flagged on a patch with fewer directions than
     *         its space
     *
     * @pre @p patch and the bases outlive the values, which keep references
     *      to them
     */
    ElementValues(const Patch<Basis, n>& patch,
                  const std::vector<std::reference_wrapper<const Basis>>& bases,
                  ValueFlags flags);

    /**
     * @brief Moves the values to an element and fills them at points of it
     *
     * The buffers are resized only when the number of points changes
     *
     * @param element Element index, in the numbering of the grid of the
     *        basis
     * @param points Points in parameter space, with size (num_points, dim)
     *
     * @pre @p element lies in the grid and every point lies inside it. With
     *      physical gradients or Hessians flagged, J is also invertible at
     *      the points
     */
    void reinit(int element, const Eigen::MatrixX<Scalar>& points);

    /**
     * @brief Replaces the values of every field by their Taylor series from
     *        the points of the last reinit() along shifts, up to a total
     *        order
     *
     * This is how the shifted boundary method expands the functions of a
     * surrogate point towards the true boundary. The gradients, tangents,
     * measures and physical gradients and Hessians stay at the points, where
     * the surrogate boundary is measured
     *
     * @param shifts Shift of each point in parameter space, with size
     *        (num_points, dim)
     * @param order Highest total order of the derivatives kept
     *
     * @pre reinit() was called, @p shifts has one row per point of it, and
     *      @p order is non-negative
     */
    void shift(const Eigen::MatrixX<Scalar>& shifts, int order)
        requires std::same_as<Basis, TensorBSpline<Scalar, dim>>;

    /**
     * @brief Adds to the values of every field a combination of their
     *        derivatives with weights per point, up to a total order, and
     *        keeps apart their combinations with further sets of weights
     *
     * This is how a shift rewritten with an equation takes the derivatives.
     * From the boundary, the combinations take them at the points of the
     * last reinit() moved by shifts, the closest points on the true
     * boundary. From the surrogate, they take them at the points themselves,
     * and the values become their Taylor series along the shifts first. A
     * trace that takes another field reads the further combinations, as the
     * velocity shifted with the Stokes equation takes the pressure gradient,
     * and reinit() drops them. The gradients, tangents, measures and
     * physical gradients and Hessians stay at the points
     *
     * @param shifts Shift of each point in parameter space, with size
     *        (num_points, dim)
     * @param weights Weight of each multi-index at each point, as
     *        TensorBSpline::combine_on_element() takes them
     * @param coupled Further sets of weights, each of the size of
     *        @p weights, whose combinations combination() gives
     * @param order Highest total order of the derivatives, and of the
     *        Taylor series
     * @param center Point the series expands about
     *
     * @pre reinit() was called, @p shifts, @p weights and every set of
     *      @p coupled have one row per point of it, and @p order is
     *      non-negative
     */
    void shift(const Eigen::MatrixX<Scalar>& shifts,
               const Eigen::MatrixX<Scalar>& weights,
               const std::vector<Eigen::MatrixX<Scalar>>& coupled, int order,
               ShiftCenter center)
        requires std::same_as<Basis, TensorBSpline<Scalar, dim>>;

    /**
     * @brief Replaces the values of every field by their derivatives across
     *        the knot lines of a direction, of the order of the degree of its
     *        basis there, in physical space
     *
     * Across a knot line of maximal continuity of a basis, these are the
     * only derivatives that jump, as a ghost penalty compares them. The
     * derivative of order p along the direction is scaled by |J^-T e_k|^p,
     * the length of the physical gradient of its parameter. On an affine map
     * its jump across the knot line is then that of the physical normal
     * derivative of order p, and with orthogonal parametric directions, as
     * on a rectangle, it is that derivative itself. The gradients, tangents,
     * measures and physical gradients and Hessians stay those of the last
     * reinit()
     *
     * @param direction Direction across the knot lines
     *
     * @pre reinit() was called, and @p direction lies in [0, dim)
     */
    void differentiate(std::size_t direction)
        requires std::same_as<Basis, TensorBSpline<Scalar, dim>>;

    /// @brief Number of fields, one per basis
    constexpr std::size_t num_fields() const noexcept
    { return fields_.size(); }

    /// @brief Number of further combinations the last shift() kept, none
    ///        after reinit()
    constexpr std::size_t num_combinations() const noexcept
    { return fields_.front().combinations.size(); }

    /**
     * @brief Combination of the derivatives of the active functions of a
     *        field with one further set of weights of the last shift(), of
     *        the size of values()
     *
     * @param field Field index, in the order of the bases
     * @param set Index of the set of weights
     *
     * @pre @p field lies in [0, num_fields()) and @p set in
     *      [0, num_combinations())
     */
    constexpr const Eigen::MatrixX<Scalar>& combination(std::size_t field,
                                                        std::size_t set) const
        noexcept
    { return fields_[field].combinations[set]; }

    /**
     * @brief Values of the active functions of a field, of size
     *        (num_active, num_points)
     *
     * @param field Field index, in the order of the bases
     *
     * @pre @p field lies in [0, num_fields())
     */
    constexpr const Eigen::MatrixX<Scalar>& values(std::size_t field = 0) const
        noexcept
    { return fields_[field].values; }

    /**
     * @brief Derivatives of the active functions of a field along each
     *        parametric direction, one matrix per direction of the size of
     *        values()
     *
     * @param field Field index, in the order of the bases
     *
     * @pre @p field lies in [0, num_fields())
     */
    constexpr const std::array<Eigen::MatrixX<Scalar>, dim>&
    gradients(std::size_t field = 0) const noexcept
    { return fields_[field].gradients; }

    /// @brief Derivatives of the map along each parametric direction, the
    ///        columns of J, one buffer of size (num_points, n) per direction
    constexpr const std::array<PointMatrix<Scalar, n>, dim>& tangents()
        const noexcept
    { return tangents_; }

    /**
     * @brief Measure of the patch at each point, the factor that turns a
     *        parametric weight into a physical one
     *
     * It is |det J| on a domain and sqrt(det(J^T J)) on a curve or surface.
     * At points of an embedded boundary it measures the patch, not the
     * boundary, whose weights need the boundary normals as well
     */
    constexpr const Eigen::VectorX<Scalar>& measures() const noexcept
    { return measures_; }

    /**
     * @brief Size h of the element at each point, the dim-th root of its
     *        measure in physical space as the map gives it there
     *
     * It is (measure x volume of the element in parameter space)^(1/dim),
     * the side of a cube of the same measure, as penalties scaled by the
     * element size need it
     */
    constexpr const Eigen::VectorX<Scalar>& sizes() const noexcept
    { return sizes_; }

    /**
     * @brief Gradients of the active functions of a field in physical
     *        space, one matrix per coordinate of the size of values()
     *
     * @param field Field index, in the order of the bases
     *
     * @pre Physical gradients or Hessians are flagged, which is not
     *      checked, and @p field lies in [0, num_fields())
     */
    constexpr const std::array<Eigen::MatrixX<Scalar>, n>&
    physical_gradients(std::size_t field = 0) const noexcept
        requires (dim == n)
    { return fields_[field].physical_gradients; }

    /**
     * @brief Second derivatives of the active functions of a field in
     *        physical space, one matrix per pair of coordinates of the size
     *        of values()
     *
     * @details The pure pairs come first, then the mixed ones in
     *          lexicographic order of their coordinates: xx, yy, xy in the
     *          plane, so that the Laplacian sums the first n
     *
     * @param field Field index, in the order of the bases
     *
     * @pre Physical Hessians are flagged, which is not checked, and
     *      @p field lies in [0, num_fields())
     */
    constexpr const std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>&
    physical_hessians(std::size_t field = 0) const noexcept
        requires (dim == n)
    { return fields_[field].physical_hessians; }

private:
    /// @brief Functions of one basis on the element and their derivatives
    struct Field
    {
        /// @brief Constructs the empty buffers of a basis, filled by
        ///        evaluate()
        explicit Field(const Basis& basis) : basis(basis)
        {
        }

        /// @brief Basis whose functions are evaluated
        const Basis& basis;

        /// @brief First active function of the basis along each direction
        std::array<int, dim> first_active{};

        /// @brief Functions of the basis active on the element
        Eigen::VectorXi actives;

        /// @brief Values of the active functions
        Eigen::MatrixX<Scalar> values;

        /// @brief Derivatives of the active functions along each direction
        std::array<Eigen::MatrixX<Scalar>, dim> gradients;

        /// @brief Second derivatives of the active functions along each pair
        ///        of directions, empty unless physical Hessians are flagged
        std::array<Eigen::MatrixX<Scalar>, dim * (dim + 1) / 2> hessians;

        /// @brief Gradients of the active functions in physical space,
        ///        empty unless flagged
        std::array<Eigen::MatrixX<Scalar>, n> physical_gradients;

        /// @brief Second derivatives of the active functions in physical
        ///        space, empty unless flagged
        std::array<Eigen::MatrixX<Scalar>, n * (n + 1) / 2>
            physical_hessians;

        /// @brief Combinations of the derivatives of the active functions
        ///        with further sets of weights, kept by shift()
        std::vector<Eigen::MatrixX<Scalar>> combinations;
    };

    /**
     * @brief Fills the values and parametric derivatives of a field at
     *        points of an element, with the second derivatives when
     *        physical Hessians are flagged
     *
     * @param field Field to fill
     * @param element Element index, in the numbering of the grid
     * @param points Points in parameter space, with size (num_points, dim)
     */
    void evaluate(Field& field, int element,
                  const Eigen::MatrixX<Scalar>& points);

    /// @brief Patch that maps the points
    const Patch<Basis, n>& patch_;

    /// @brief Optional values to compute
    ValueFlags flags_;

    /// @brief Points of the element in parameter space
    Eigen::MatrixX<Scalar> points_;

    /// @brief Functions of each basis, in the order of the bases
    std::vector<Field> fields_;

    /// @brief First field whose basis is the patch's own, whose functions
    ///        then give the map as well
    std::optional<std::size_t> map_field_;

    /// @brief Functions of the patch, which give the map when no basis is
    ///        the patch's
    Field map_;

    /// @brief Derivatives of the map along each direction
    std::array<PointMatrix<Scalar, n>, dim> tangents_;

    /// @brief Second derivatives of the map along each pair of directions,
    ///        empty unless physical Hessians are flagged
    std::array<PointMatrix<Scalar, n>, dim * (dim + 1) / 2> second_tangents_;

    /// @brief Measure of the patch at each point
    Eigen::VectorX<Scalar> measures_;

    /// @brief Size of the element at each point
    Eigen::VectorX<Scalar> sizes_;
};

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ELEMENT_VALUES_HPP
