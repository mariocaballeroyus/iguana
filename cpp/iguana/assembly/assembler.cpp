/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "assembler.hpp"

#include<stdexcept>
#include<utility>

namespace iguana
{

template<int d, std::floating_point T>
Assembler<d, T>::Assembler(TensorDomain<d, T> domain, Patch<d, T> patch,
                           GaussLegendre<d, T> rule,
                           std::unique_ptr<Element<d, T>> element)
    : domain_(std::move(domain)), patch_(std::move(patch)),
      rule_(std::move(rule)), element_(std::move(element))
{
    if (!element_)
        throw std::invalid_argument("Assembler: "
                                    "the element must not be null");
}

template<int d, std::floating_point T>
void Assembler<d, T>::assemble(Eigen::SparseMatrix<T>& jacobian,
                               Eigen::VectorX<T>& residual) const
{
    const TensorProductBSpline<d, T>& basis = domain_.basis();
    const int num_active = basis.num_active();

    ElementData<d, T> data(domain_, patch_, rule_,
                           element_->updates());

    // Local buffers, zeroed per element and added to by the integrand
    Eigen::MatrixX<T> local_jacobian(num_active, num_active);
    Eigen::VectorX<T> local_residual(num_active);

    std::vector<Eigen::Triplet<T>> triplets;
    triplets.reserve(static_cast<std::size_t>(domain_.num_elements())
                     * static_cast<std::size_t>(num_active * num_active));

    residual = Eigen::VectorX<T>::Zero(basis.num_functions());

    for (const auto& element : domain_) {
        data.reinit(element);

        local_jacobian.setZero();
        local_residual.setZero();

        element_->residual(data, local_residual);
        element_->jacobian(data, local_jacobian);

        // Scattered by the actives of the element
        residual(data.actives) += local_residual;

        for (int i = 0; i < num_active; ++i)
            for (int j = 0; j < num_active; ++j)
                triplets.emplace_back(data.actives[i], data.actives[j],
                                      local_jacobian(i, j));
    }

    jacobian.resize(basis.num_functions(), basis.num_functions());
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

template class Assembler<1, double>;
template class Assembler<2, double>;
template class Assembler<3, double>;

} // namespace iguana
