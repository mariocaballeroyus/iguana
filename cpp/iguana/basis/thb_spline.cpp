/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "thb_spline.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "iguana/basis/bspline.hpp"
#include "iguana/grid/knot_vector.hpp"
#include "iguana/grid/tensor_grid.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Open knot vector of a degree on the knots of another, with its
 *        interior knots and its ends repeated once per degree plus one
 *
 * @throws std::invalid_argument If the knot vector is not open, with its
 *         first p + 1 knots at the start of its domain and its last p + 1
 *         at its end
 */
template<std::floating_point T>
KnotVector<T> with_degree(const KnotVector<T>& knots, int degree)
{
    const std::vector<T>& values = knots.values();
    const std::size_t repeats = static_cast<std::size_t>(knots.degree()) + 1;
    const T start = knots.domain_start();
    const T end = knots.domain_end();

    if (!std::all_of(values.begin(), values.begin() + repeats,
                     [start](T knot) { return knot == start; })
        || !std::all_of(values.end() - repeats, values.end(),
                        [end](T knot) { return knot == end; }))
        throw std::invalid_argument("THBSpline: "
                                    "the coarse knot vectors must be open");

    const std::size_t count = static_cast<std::size_t>(degree) + 1;
    std::vector<T> open(count, start);
    open.insert(open.end(), values.begin() + repeats, values.end() - repeats);
    open.insert(open.end(), count, end);

    return KnotVector<T>(degree, std::move(open));
}

/// @brief Tensor basis of given degrees on the elements of a grid
template<std::floating_point T, std::size_t d, std::size_t... direction>
TensorBSpline<T, d> basis_on(const TensorGrid<T, d>& grid,
                             const std::array<int, d>& degrees,
                             std::index_sequence<direction...>)
{
    return TensorBSpline<T, d>(std::array<BSpline<T>, d>{BSpline<T>(
        with_degree(grid.knots(direction), degrees[direction]))...});
}

/// @brief Number of functions of each direction of a tensor basis
template<std::floating_point T, std::size_t d>
std::array<int, d> function_counts(const TensorBSpline<T, d>& basis) noexcept
{
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] = basis.axis(direction).num_functions();

    return counts;
}

/// @brief Number of elements of each direction of a grid
template<std::floating_point T, std::size_t d>
std::array<int, d> element_counts(const TensorGrid<T, d>& grid) noexcept
{
    std::array<int, d> counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        counts[direction] = grid.knots(direction).num_elements();

    return counts;
}

/// @brief Whether a sorted vector holds a value
bool holds(const std::vector<int>& sorted, int value) noexcept
{
    return std::ranges::binary_search(sorted, value);
}

/// @brief Position of a value in a sorted vector holding it
int position_in(const std::vector<int>& sorted, int value) noexcept
{
    return static_cast<int>(
        std::distance(sorted.begin(), std::ranges::lower_bound(sorted, value)));
}

} // namespace

template<std::floating_point T, std::size_t d>
THBSpline<T, d>::THBSpline(HierarchicalGrid<T, d> grid,
                           std::array<int, d> degrees)
    : grid_(std::move(grid)),
      degrees_(degrees)
{
    for (const int degree : degrees_) {
        if (degree < 0 || degree > BSpline<T>::max_degree)
            throw std::invalid_argument("THBSpline: "
                                        "the degrees must lie in [0, "
                                        "BSpline::max_degree]");
    }

    for (int level = 0; level < grid_.num_levels(); ++level)
        levels_.push_back(basis_on(grid_.level(level), degrees_,
                                   std::make_index_sequence<d>{}));

    // The band of fine functions each coarse function refines into, read
    // off the refinement matrix of the knot vectors
    for (int level = 1; level < num_levels(); ++level) {
        std::array<Refinement, d>& refinement = refinements_.emplace_back();

        for (std::size_t direction = 0; direction < d; ++direction) {
            const Eigen::MatrixX<T> matrix = refinement_matrix(
                levels_[level - 1].axis(direction).knots(),
                levels_[level].axis(direction).knots());
            const Eigen::Index coarse = matrix.cols();
            Eigen::VectorXi first(coarse);
            Eigen::VectorXi last(coarse);

            for (Eigen::Index function = 0; function < coarse; ++function) {
                Eigen::Index row = 0;

                while (matrix(row, function) == 0)
                    ++row;

                first(function) = static_cast<int>(row);
                row = matrix.rows() - 1;

                while (matrix(row, function) == 0)
                    --row;

                last(function) = static_cast<int>(row);
            }

            const int width = (last - first).maxCoeff() + 1;
            Eigen::MatrixX<T> band = Eigen::MatrixX<T>::Zero(width, coarse);

            for (Eigen::Index function = 0; function < coarse; ++function)
                band.col(function).head(last(function) - first(function) + 1) =
                    matrix.col(function).segment(
                        first(function), last(function) - first(function) + 1);

            refinement[direction] = {std::move(first), std::move(band)};
        }
    }

    // Kraft's selection: of the functions non-zero on the domain of a
    // level, those whose support lies in the domain, every element of it
    // refined at least to the level, but not in the domain of the next,
    // some element of it active on the level
    offsets_.assign(1, 0);

    for (int level = 0; level < num_levels(); ++level) {
        const TensorBSpline<T, d>& basis = levels_[level];
        const std::vector<int>& domain = grid_.domain(level);
        const std::vector<int>& active = grid_.active_elements(level);
        const std::array<int, d> counts = function_counts(basis);
        const std::array<int, d> elements = element_counts(basis.grid());

        std::vector<int> candidates;
        Eigen::VectorXi actives;

        for (const int element : domain) {
            basis.active_on_element(element, actives);
            candidates.insert(candidates.end(), actives.begin(), actives.end());
        }

        std::ranges::sort(candidates);
        const auto repeats = std::ranges::unique(candidates);
        candidates.erase(repeats.begin(), repeats.end());

        std::vector<int>& selected = active_.emplace_back();

        for (const int function : candidates) {
            const std::array<int, d> index = unflatten(function, counts);
            std::array<int, d> start{};
            std::array<int, d> extent{};

            for (std::size_t direction = 0; direction < d; ++direction) {
                const std::array<int, 2> support =
                    basis.axis(direction).knots().support(index[direction]);
                start[direction] = support[0];
                extent[direction] = support[1] - support[0] + 1;
            }

            bool inside = true;
            bool meets_active = false;
            std::array<int, d> offset{};

            do {
                std::array<int, d> element = start;

                for (std::size_t direction = 0; direction < d; ++direction)
                    element[direction] += offset[direction];

                const int flat = flatten(element, elements);
                inside = inside && holds(domain, flat);
                meets_active = meets_active || holds(active, flat);
            } while (inside && next_lexicographic(offset, extent));

            if (inside && meets_active)
                selected.push_back(function);
        }

        offsets_.push_back(offsets_.back()
                           + static_cast<int>(selected.size()));
    }
}

template<std::floating_point T, std::size_t d>
std::vector<typename THBSpline<T, d>::Selection>
THBSpline<T, d>::selections(int element) const
{
    const int top = grid_.level_of(element);
    const std::array<int, d> index = unflatten(
        grid_.level_index(element), element_counts(grid_.level(top)));

    std::vector<Selection> selected;
    Eigen::VectorXi actives;

    for (int level = 0; level <= top; ++level) {
        // The ancestor of the element on the level, i / 2 per level up
        std::array<int, d> ancestor = index;

        for (int& axis : ancestor)
            axis >>= top - level;

        const TensorBSpline<T, d>& basis = levels_[level];
        const int flat = flatten(ancestor, element_counts(basis.grid()));
        basis.active_on_element(flat, actives);

        Selection& selection = selected.emplace_back();
        selection.first_active = basis.first_active(flat);

        for (int local = 0; local < actives.size(); ++local) {
            const int function = actives(local);

            if (holds(active_[level], function)) {
                selection.local.push_back(local);
                selection.global.push_back(
                    offset(level) + position_in(active_[level], function));
            }
        }
    }

    return selected;
}

template<std::floating_point T, std::size_t d>
typename THBSpline<T, d>::Local THBSpline<T, d>::localize(int element) const
{
    const std::vector<Selection> selected = selections(element);
    const int num_active = levels_[0].num_active();
    std::array<int, d> active_counts{};

    for (std::size_t direction = 0; direction < d; ++direction)
        active_counts[direction] = degrees_[direction] + 1;

    // The columns combine the functions of a level non-zero on the
    // ancestor, the selected ones as themselves
    Eigen::MatrixX<T> combination(num_active, 0);

    for (std::size_t level = 0; level < selected.size(); ++level) {
        const Selection& selection = selected[level];

        if (level > 0) {
            // The functions of the level before written in those of this
            // level, the product of the bands of the directions
            const Selection& coarse = selected[level - 1];
            Eigen::MatrixX<T> refinement(num_active, num_active);
            std::array<int, d> fine{};

            do {
                std::array<int, d> from{};

                do {
                    T product = 1;

                    for (std::size_t direction = 0; direction < d;
                         ++direction) {
                        const Refinement& band =
                            refinements_[level - 1][direction];
                        const int function =
                            coarse.first_active[direction] + from[direction];
                        const int row = selection.first_active[direction]
                                        + fine[direction]
                                        - band.first(function);

                        product *= row >= 0 && row < band.coefficients.rows()
                                       ? band.coefficients(row, function)
                                       : T{0};
                    }

                    refinement(flatten(fine, active_counts),
                               flatten(from, active_counts)) = product;
                } while (next_lexicographic(from, active_counts));
            } while (next_lexicographic(fine, active_counts));

            combination = refinement * combination;

            // Truncation drops the components along the functions of the
            // level that belong to the basis
            for (const int local : selection.local)
                combination.row(local).setZero();
        }

        const Eigen::Index columns = combination.cols();
        combination.conservativeResize(
            Eigen::NoChange, columns + std::ssize(selection.local));

        for (std::size_t k = 0; k < selection.local.size(); ++k) {
            combination.col(columns + static_cast<Eigen::Index>(k)).setZero();
            combination(selection.local[k],
                        columns + static_cast<Eigen::Index>(k)) = 1;
        }
    }

    // A coarse function the truncation leaves nothing of on the element
    // has an exactly zero column, and is left out
    Local local{selected.back().first_active, {}, {}};
    std::vector<int> kept;
    Eigen::Index column = 0;

    for (const Selection& selection : selected) {
        for (const int function : selection.global) {
            if (!combination.col(column).isZero(0))
                kept.push_back(function);

            ++column;
        }
    }

    local.global = Eigen::Map<const Eigen::VectorXi>(kept.data(),
                                                     std::ssize(kept));
    local.extraction.resize(std::ssize(kept), num_active);
    column = 0;
    Eigen::Index row = 0;

    for (const Selection& selection : selected) {
        for (std::size_t k = 0; k < selection.global.size(); ++k, ++column) {
            if (!combination.col(column).isZero(0))
                local.extraction.row(row++) = combination.col(column);
        }
    }

    return local;
}

template<std::floating_point T, std::size_t d>
void THBSpline<T, d>::active_on_element(int element,
                                        Eigen::VectorXi& actives) const
{
    actives = localize(element).global;
}

template<std::floating_point T, std::size_t d>
void THBSpline<T, d>::eval_on_element(int element,
                                      const Eigen::MatrixX<T>& points,
                                      Eigen::MatrixX<T>& values) const
{
    const Local local = localize(element);
    Eigen::MatrixX<T> tensor;

    levels_[grid_.level_of(element)].eval_on_element(local.first_active,
                                                     points, tensor);
    values = local.extraction * tensor;
}

template<std::floating_point T, std::size_t d>
void THBSpline<T, d>::grad_on_element(
    int element, const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values,
    std::array<Eigen::MatrixX<T>, d>& gradients) const
{
    const Local local = localize(element);
    Eigen::MatrixX<T> tensor;
    std::array<Eigen::MatrixX<T>, d> tensor_gradients;

    levels_[grid_.level_of(element)].grad_on_element(
        local.first_active, points, tensor, tensor_gradients);
    values = local.extraction * tensor;

    for (std::size_t direction = 0; direction < d; ++direction)
        gradients[direction] = local.extraction * tensor_gradients[direction];
}

template<std::floating_point T, std::size_t d>
void THBSpline<T, d>::hess_on_element(
    int element, const Eigen::MatrixX<T>& points, Eigen::MatrixX<T>& values,
    std::array<Eigen::MatrixX<T>, d>& gradients,
    std::array<Eigen::MatrixX<T>, d * (d + 1) / 2>& hessians) const
{
    const Local local = localize(element);
    Eigen::MatrixX<T> tensor;
    std::array<Eigen::MatrixX<T>, d> tensor_gradients;
    std::array<Eigen::MatrixX<T>, d * (d + 1) / 2> tensor_hessians;

    levels_[grid_.level_of(element)].hess_on_element(
        local.first_active, points, tensor, tensor_gradients,
        tensor_hessians);
    values = local.extraction * tensor;

    for (std::size_t direction = 0; direction < d; ++direction)
        gradients[direction] = local.extraction * tensor_gradients[direction];

    for (std::size_t pair = 0; pair < d * (d + 1) / 2; ++pair)
        hessians[pair] = local.extraction * tensor_hessians[pair];
}

template<std::floating_point T, std::size_t d>
Eigen::MatrixX<T> THBSpline<T, d>::coefficients(
    const Eigen::MatrixX<T>& coarse) const
{
    if (coarse.rows() != levels_[0].num_functions())
        throw std::invalid_argument("THBSpline: "
                                    "the coefficients must be one per "
                                    "function of the coarse basis");

    Eigen::MatrixX<T> result(num_functions(), coarse.cols());
    Eigen::MatrixX<T> current = coarse;

    for (int level = 0; level < num_levels(); ++level) {
        if (level > 0) {
            // Refine along each direction in turn: with the first direction
            // running fastest, the coefficients split into blocks of the
            // directions before it, by the function of the direction, by
            // those after it, and each block of the directions before
            // refines as a matrix with the functions of the direction as
            // columns
            std::array<int, d> counts = function_counts(levels_[level - 1]);
            const std::array<int, d> fine_counts =
                function_counts(levels_[level]);

            for (std::size_t direction = 0; direction < d; ++direction) {
                const Refinement& band = refinements_[level - 1][direction];
                const Eigen::Index inner = std::accumulate(
                    counts.begin(), counts.begin() + direction, Eigen::Index{1},
                    std::multiplies<>{});
                const Eigen::Index outer = std::accumulate(
                    counts.begin() + direction + 1, counts.end(),
                    Eigen::Index{1}, std::multiplies<>{});
                const Eigen::Index n = counts[direction];
                const Eigen::Index m = fine_counts[direction];

                Eigen::MatrixX<T> refined =
                    Eigen::MatrixX<T>::Zero(inner * m * outer, current.cols());

                for (Eigen::Index column = 0; column < current.cols();
                     ++column) {
                    for (Eigen::Index block = 0; block < outer; ++block) {
                        const Eigen::Map<const Eigen::MatrixX<T>> from(
                            current.col(column).data() + block * inner * n,
                            inner, n);
                        Eigen::Map<Eigen::MatrixX<T>> to(
                            refined.col(column).data() + block * inner * m,
                            inner, m);

                        for (Eigen::Index function = 0; function < n;
                             ++function) {
                            const Eigen::Index width = band.coefficients.rows();
                            to.middleCols(band.first(function), width) +=
                                from.col(function)
                                * band.coefficients.col(function).transpose();
                        }
                    }
                }

                current = std::move(refined);
                counts[direction] = fine_counts[direction];
            }
        }

        const std::vector<int>& selected = active_[static_cast<std::size_t>(
            level)];

        for (std::size_t k = 0; k < selected.size(); ++k)
            result.row(offset(level) + static_cast<Eigen::Index>(k)) =
                current.row(selected[k]);
    }

    return result;
}

template<std::floating_point T, std::size_t d>
Eigen::VectorXi THBSpline<T, d>::boundary_functions(
    std::size_t direction) const
{
    std::vector<int> result;

    for (int level = 0; level < num_levels(); ++level) {
        const std::array<int, d> counts = function_counts(levels_[level]);
        const std::vector<int>& selected = active_[static_cast<std::size_t>(
            level)];

        for (std::size_t k = 0; k < selected.size(); ++k) {
            const int index = unflatten(selected[k], counts)[direction];

            if (index == 0 || index == counts[direction] - 1)
                result.push_back(offset(level) + static_cast<int>(k));
        }
    }

    return Eigen::Map<const Eigen::VectorXi>(result.data(),
                                             std::ssize(result));
}

template<std::floating_point T, std::size_t d>
bool THBSpline<T, d>::operator==(const THBSpline& other) const
{
    return grid_ == other.grid_ && degrees_ == other.degrees_;
}

template class THBSpline<double, 2>;
template class THBSpline<double, 3>;

} // namespace iguana
