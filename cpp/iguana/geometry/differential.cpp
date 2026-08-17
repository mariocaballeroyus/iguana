/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "differential.hpp"

namespace iguana
{

template<std::size_t d, std::floating_point T>
void metric(const std::array<Eigen::MatrixX3<T>, d>& tangents,
            Eigen::MatrixX<T>& g)
{
    constexpr int slots = static_cast<int>(d * (d + 1) / 2);

    // Output buffer, reused across elements
    g.resize(tangents[0].rows(), slots);

    // The diagonal first, in direction order
    for (std::size_t k = 0; k < d; ++k)
        g.col(static_cast<int>(k))
            = tangents[k].rowwise().squaredNorm();

    // Then the mixed pairs, in the lexicographic order of the pairs
    int slot = static_cast<int>(d);

    for (std::size_t k = 0; k < d; ++k)
        for (std::size_t l = k + 1; l < d; ++l)
            g.col(slot++) = (tangents[k].array() * tangents[l].array())
                                .rowwise()
                                .sum();
}

template<std::size_t d, std::floating_point T>
void measure(const Eigen::MatrixX<T>& g,
             Eigen::VectorX<T>& measures)
{
    // Output buffer, reused across elements
    measures.resize(g.rows());

    if constexpr (d == 1) {
        measures = g.col(0).cwiseSqrt();
    }
    else if constexpr (d == 2) {
        // det g of the symmetric 2 x 2
        const Eigen::ArrayX<T> det = g.col(0).array() * g.col(1).array()
                                     - g.col(2).array().square();

        measures = det.sqrt();
    }
    else {
        // det g of the symmetric 3 x 3, by cofactors of the first row
        const Eigen::ArrayX<T> det
            = g.col(0).array()
                  * (g.col(1).array() * g.col(2).array()
                     - g.col(5).array().square())
              - g.col(3).array()
                    * (g.col(3).array() * g.col(2).array()
                       - g.col(5).array() * g.col(4).array())
              + g.col(4).array()
                    * (g.col(3).array() * g.col(5).array()
                       - g.col(1).array() * g.col(4).array());

        measures = det.sqrt();
    }
}

template<std::size_t d, std::floating_point T>
void metric_inverse(const Eigen::MatrixX<T>& g,
                    Eigen::MatrixX<T>& inverse)
{
    // Output buffer, reused across elements
    inverse.resize(g.rows(), g.cols());

    if constexpr (d == 1) {
        inverse.col(0) = g.col(0).cwiseInverse();
    }
    else if constexpr (d == 2) {
        // One reciprocal spares a division per component
        const Eigen::ArrayX<T> inv_det
            = (g.col(0).array() * g.col(1).array()
               - g.col(2).array().square())
                  .inverse();

        inverse.col(0) = g.col(1).array() * inv_det;
        inverse.col(1) = g.col(0).array() * inv_det;
        inverse.col(2) = -g.col(2).array() * inv_det;
    }
    else {
        // The adjugate of the symmetric 3 x 3 over its determinant
        const Eigen::ArrayX<T> det
            = g.col(0).array()
                  * (g.col(1).array() * g.col(2).array()
                     - g.col(5).array().square())
              - g.col(3).array()
                    * (g.col(3).array() * g.col(2).array()
                       - g.col(5).array() * g.col(4).array())
              + g.col(4).array()
                    * (g.col(3).array() * g.col(5).array()
                       - g.col(1).array() * g.col(4).array());

        inverse.col(0) = (g.col(1).array() * g.col(2).array()
                          - g.col(5).array().square())
                         / det;
        inverse.col(1) = (g.col(0).array() * g.col(2).array()
                          - g.col(4).array().square())
                         / det;
        inverse.col(2) = (g.col(0).array() * g.col(1).array()
                          - g.col(3).array().square())
                         / det;
        inverse.col(3) = (g.col(4).array() * g.col(5).array()
                          - g.col(3).array() * g.col(2).array())
                         / det;
        inverse.col(4) = (g.col(3).array() * g.col(5).array()
                          - g.col(1).array() * g.col(4).array())
                         / det;
        inverse.col(5) = (g.col(3).array() * g.col(4).array()
                          - g.col(0).array() * g.col(5).array())
                         / det;
    }
}

template void metric(const std::array<Eigen::MatrixX3<double>, 1>&,
                     Eigen::MatrixXd&);
template void metric(const std::array<Eigen::MatrixX3<double>, 2>&,
                     Eigen::MatrixXd&);
template void metric(const std::array<Eigen::MatrixX3<double>, 3>&,
                     Eigen::MatrixXd&);

template void measure<1, double>(const Eigen::MatrixXd&,
                                 Eigen::VectorXd&);
template void measure<2, double>(const Eigen::MatrixXd&,
                                 Eigen::VectorXd&);
template void measure<3, double>(const Eigen::MatrixXd&,
                                 Eigen::VectorXd&);

template void metric_inverse<1, double>(const Eigen::MatrixXd&,
                                        Eigen::MatrixXd&);
template void metric_inverse<2, double>(const Eigen::MatrixXd&,
                                        Eigen::MatrixXd&);
template void metric_inverse<3, double>(const Eigen::MatrixXd&,
                                        Eigen::MatrixXd&);

} // namespace iguana
