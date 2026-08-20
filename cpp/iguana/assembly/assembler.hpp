/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#ifndef IGUANA_ASSEMBLY_ASSEMBLER_HPP
#define IGUANA_ASSEMBLY_ASSEMBLER_HPP

#include<concepts>
#include<memory>

#include<Eigen/Sparse>

#include "iguana/assembly/element.hpp"

namespace iguana
{

/**
 * @brief Assembles the global system of a problem over its domain.
 *
 * The assembler owns everything problem-independent: it walks the
 * elements of the domain, refills an ElementData with the quantities
 * its integrand requests, and scatters the local contributions into
 * the global system by the actives. The physics lives in the
 * integrand it intakes: one Element over the domain, and later the
 * Conditions over facets.
 *
 * @tparam d Number of parametric directions.
 * @tparam T Floating-point type of the system.
 */
template<int d, std::floating_point T>
class Assembler
{
public:
    /**
     * @brief Constructs the assembler of a problem.
     *
     * @param domain The domain whose elements are integrated.
     * @param patch The geometry of the domain. Its basis is the basis
     *        of the unknown, the isoparametric choice.
     * @param rule The quadrature rule applied on every element.
     * @param element The domain integrand of the problem.
     *
     * @throws std::invalid_argument If the element is null.
     *
     * @pre The domain and the patch hold the same basis. It is not
     *      checked.
     */
    Assembler(TensorDomain<d, T> domain, Patch<d, T> patch,
              GaussLegendre<d, T> rule,
              std::unique_ptr<Element<d, T>> element);

    /// @brief The domain whose elements are integrated.
    constexpr const TensorDomain<d, T>& domain() const noexcept
    { return domain_; }

    /**
     * @brief Assembles the jacobian and the residual of the problem.
     *
     * Every integrand adds its local contributions, and the solve of
     * \f$ J \, \Delta u = -r \f$ is the step of a Newton iteration,
     * or the whole linear system at zero state.
     *
     * @param jacobian Output system jacobian, of size
     *        (num_functions,num_functions), overwritten.
     * @param residual Output residual, of size (num_functions,),
     *        overwritten.
     */
    void assemble(Eigen::SparseMatrix<T>& jacobian,
                  Eigen::VectorX<T>& residual) const;

private:
    /// @brief The domain whose elements are integrated.
    TensorDomain<d, T> domain_;

    /// @brief The geometry of the domain.
    Patch<d, T> patch_;

    /// @brief The quadrature rule applied on every element.
    GaussLegendre<d, T> rule_;

    /// @brief The domain integrand of the problem.
    std::unique_ptr<Element<d, T>> element_;
};

extern template class Assembler<1, double>;
extern template class Assembler<2, double>;
extern template class Assembler<3, double>;

} // namespace iguana

#endif // IGUANA_ASSEMBLY_ASSEMBLER_HPP
