/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "function_space.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace iguana
{

namespace
{

/// @brief Degrees of freedom of the functions active on the cells that are
///        not outside, numbered in increasing function index
template<std::floating_point T, std::size_t d>
DofMap standard_dof_map(const TensorBSpline<T, d>& basis,
                        const Embedding<T, d>& embedding)
{
    const int num_elements = basis.domain().num_elements();

    if (embedding.num_elements() != num_elements)
        throw std::invalid_argument("FunctionSpace: "
                                    "the embedding must have one cell type "
                                    "per element");

    // Each cell that is not outside lists its active functions, an outside
    // cell none
    Eigen::VectorXi offsets = Eigen::VectorXi::Zero(num_elements + 1);
    std::vector<int> functions;
    Eigen::VectorXi actives;

    for (int element = 0; element < num_elements; ++element) {
        if (embedding.cell_type(element) != CellType::outside) {
            basis.active_on_element(element, actives);
            functions.insert(functions.end(), actives.begin(), actives.end());
        }
        offsets(element + 1) = static_cast<int>(functions.size());
    }

    // Mark the listed functions with zero, then number them in increasing
    // function index, so that those active on outside cells alone keep -1
    Eigen::VectorXi numbering =
        Eigen::VectorXi::Constant(basis.num_functions(), -1);
    numbering(functions).setZero();

    int num_dofs = 0;

    for (int& dof : numbering) {
        if (dof == 0)
            dof = num_dofs++;
    }

    Eigen::VectorXi dofs = numbering(functions);

    return DofMap(num_dofs, std::move(offsets), std::move(dofs));
}

} // namespace

template<std::floating_point T, std::size_t d>
FunctionSpace<T, d>::FunctionSpace(TensorBSpline<T, d> basis,
                                   const Embedding<T, d>& embedding)
    : basis_(std::move(basis)),
      dof_map_(standard_dof_map(basis_, embedding))
{
}

template class FunctionSpace<double, 1>;
template class FunctionSpace<double, 2>;
template class FunctionSpace<double, 3>;

} // namespace iguana
