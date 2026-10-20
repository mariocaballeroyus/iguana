/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "mixed_function_space.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace iguana
{

template<typename Basis>
MixedFunctionSpace<Basis>::MixedFunctionSpace(
    std::vector<std::reference_wrapper<const FunctionSpace<Basis>>> fields)
    : fields_(std::move(fields))
{
    if (fields_.empty())
        throw std::invalid_argument("MixedFunctionSpace: "
                                    "the space needs at least one field");

    // One element index locates the points in every basis
    const std::array<std::vector<Scalar>, dimension> lines =
        fields_.front().get().basis().grid().lines();

    offsets_.reserve(fields_.size() + 1);
    offsets_.push_back(0);

    for (const FunctionSpace<Basis>& field : fields_) {
        if (field.basis().grid().lines() != lines)
            throw std::invalid_argument("MixedFunctionSpace: "
                                        "every field must have the elements "
                                        "of the first");

        offsets_.push_back(offsets_.back() + field.num_dofs());
    }
}

template<typename Basis>
std::vector<std::reference_wrapper<const Basis>>
MixedFunctionSpace<Basis>::bases() const
{
    std::vector<std::reference_wrapper<const Basis>> bases;
    bases.reserve(fields_.size());

    for (const FunctionSpace<Basis>& field : fields_)
        bases.push_back(std::cref(field.basis()));

    return bases;
}

template<typename Basis>
void MixedFunctionSpace<Basis>::dofs_on_element(int element,
                                                std::vector<int>& dofs) const
{
    std::vector<int> field_dofs;
    dofs.clear();

    for (std::size_t field = 0; field < fields_.size(); ++field) {
        fields_[field].get().dofs_on_element(element, field_dofs);

        for (const int dof : field_dofs)
            dofs.push_back(offsets_[field] + dof);
    }
}

template class MixedFunctionSpace<TensorBSpline<double, 1>>;
template class MixedFunctionSpace<TensorBSpline<double, 2>>;
template class MixedFunctionSpace<TensorBSpline<double, 3>>;
template class MixedFunctionSpace<TensorNURBS<double, 1>>;
template class MixedFunctionSpace<TensorNURBS<double, 2>>;
template class MixedFunctionSpace<TensorNURBS<double, 3>>;

} // namespace iguana
