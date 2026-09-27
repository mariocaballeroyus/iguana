/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "gauss_legendre.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iguana
{

// Rules of C. F. Gauss, Comment. Soc. Reg. Sci. Gotting. Recent. 3 (1816)
// 39-76, whose nodes are the roots of the Legendre polynomial P_n and whose
// weights are 2 / ((1 - x^2) P_n'(x)^2), rounded once to double
template<std::floating_point T, std::size_t d>
std::array<std::vector<T>, 2>
GaussLegendre<T, d>::tabulated_rule(int num_points)
{
    switch (num_points) {
    case 1: {
        // Gauss-Legendre, 1 point
        std::vector<T> x = {0.0};
        std::vector<T> w = {2.0};
        return {std::move(x), std::move(w)};
    }
    case 2: {
        // Gauss-Legendre, 2 points
        std::vector<T> x = {-0.5773502691896257, 0.5773502691896257};
        std::vector<T> w = {1.0, 1.0};
        return {std::move(x), std::move(w)};
    }
    case 3: {
        // Gauss-Legendre, 3 points
        std::vector<T> x = {-0.7745966692414834, 0.0, 0.7745966692414834};
        std::vector<T> w
            = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
        return {std::move(x), std::move(w)};
    }
    case 4: {
        // Gauss-Legendre, 4 points
        std::vector<T> x
            = {-0.8611363115940526, -0.33998104358485626, 0.33998104358485626,
               0.8611363115940526};
        std::vector<T> w
            = {0.34785484513745385, 0.6521451548625461, 0.6521451548625461,
               0.34785484513745385};
        return {std::move(x), std::move(w)};
    }
    case 5: {
        // Gauss-Legendre, 5 points
        std::vector<T> x
            = {-0.906179845938664, -0.5384693101056831, 0.0,
               0.5384693101056831, 0.906179845938664};
        std::vector<T> w
            = {0.23692688505618908, 0.47862867049936647, 0.5688888888888889,
               0.47862867049936647, 0.23692688505618908};
        return {std::move(x), std::move(w)};
    }
    case 6: {
        // Gauss-Legendre, 6 points
        std::vector<T> x
            = {-0.932469514203152, -0.6612093864662645, -0.2386191860831969,
               0.2386191860831969, 0.6612093864662645,  0.932469514203152};
        std::vector<T> w
            = {0.17132449237917036, 0.3607615730481386, 0.46791393457269104,
               0.46791393457269104, 0.3607615730481386, 0.17132449237917036};
        return {std::move(x), std::move(w)};
    }
    case 7: {
        // Gauss-Legendre, 7 points
        std::vector<T> x
            = {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
               0.0,                 0.4058451513773972,  0.7415311855993945,
               0.9491079123427585};
        std::vector<T> w
            = {0.1294849661688697, 0.27970539148927664, 0.3818300505051189,
               0.4179591836734694, 0.3818300505051189,  0.27970539148927664,
               0.1294849661688697};
        return {std::move(x), std::move(w)};
    }
    case 8: {
        // Gauss-Legendre, 8 points
        std::vector<T> x
            = {-0.9602898564975363, -0.7966664774136267, -0.525532409916329,
               -0.1834346424956498, 0.1834346424956498,  0.525532409916329,
               0.7966664774136267,  0.9602898564975363};
        std::vector<T> w
            = {0.10122853629037626, 0.22238103445337448, 0.31370664587788727,
               0.362683783378362,   0.362683783378362,   0.31370664587788727,
               0.22238103445337448, 0.10122853629037626};
        return {std::move(x), std::move(w)};
    }
    default:
        throw std::invalid_argument("GaussLegendre: "
                                    "the number of points per direction "
                                    "must lie in [1, max_points]");
    }
}

template std::array<std::vector<double>, 2>
GaussLegendre<double, 1>::tabulated_rule(int);
template std::array<std::vector<double>, 2>
GaussLegendre<double, 2>::tabulated_rule(int);
template std::array<std::vector<double>, 2>
GaussLegendre<double, 3>::tabulated_rule(int);

} // namespace iguana
