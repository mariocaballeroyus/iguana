/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <vector>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

namespace
{

using Basis = iguana::TensorBSpline<double, 2>;

/// @brief Rectangle [0, 2] x [0, 3] with quadratic elements over uneven
///        spans, whose control points are the scaled Greville points, so
///        that the map scales each axis by its side
iguana::Patch<Basis, 2> rectangle()
{
    const iguana::KnotVector<double> along_x(2, {0., 0., 0., .4, 1., 1., 1.});
    const iguana::KnotVector<double> along_y(2, {0., 0., 0., 1., 1., 1.});
    const Basis basis{iguana::TensorGrid<double, 2>({along_x, along_y})};

    iguana::PointMatrix<double, 2> points(basis.num_functions(), 2);
    const std::vector<double>& t = along_x.values();
    const std::vector<double>& s = along_y.values();

    // Greville points, the first direction running fastest
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 4; ++i)
            points.row(i + 4 * j) << 2. * (t[i + 1] + t[i + 2]) / 2.,
                                     3. * (s[j + 1] + s[j + 2]) / 2.;
    }

    return {basis, points};
}

} // namespace

TEST_CASE("The field trace of the Poisson element gives the field",
          "[poisson_traces]")
{
    using Element = iguana::PoissonElement<Basis, 2>;

    const iguana::Patch<Basis, 2> patch = rectangle();
    const Element::U trace;
    iguana::ElementValues<Basis, 2> values(patch, trace.flags());

    // Points of the second element, as a boundary through it would have
    Eigen::MatrixXd points(3, 2);
    points << .5, .1,
              .7, .6,
              .9, .95;
    values.reinit(1, points);

    Eigen::MatrixXd b;
    trace.local_trace(values, b);

    // The coefficients of a linear field are its values at the control
    // points, so the trace gives the field at the points
    Eigen::VectorXi actives;
    patch.basis().active_on_element(1, actives);

    const Eigen::Vector2d gradient(1., -2.);
    const Eigen::VectorXd u =
        patch.coefficients()(actives, Eigen::placeholders::all) * gradient;

    iguana::PointMatrix<double, 2> positions;
    patch.position_on_element(actives, values.values(), positions);

    REQUIRE((b.transpose() * u - positions * gradient).cwiseAbs().maxCoeff()
            < 1e-14);
}

TEST_CASE("The flux of the Poisson element gives the normal derivative",
          "[poisson_traces]")
{
    using Element = iguana::PoissonElement<Basis, 2>;

    const iguana::Patch<Basis, 2> patch = rectangle();
    const Element::Q flux;
    iguana::ElementValues<Basis, 2> values(patch, flux.flags());

    // Points of the second element with oblique unit normals, as a curved
    // boundary through it would have
    Eigen::MatrixXd points(3, 2);
    points << .5, .1,
              .7, .6,
              .9, .95;
    values.reinit(1, points);

    iguana::PointMatrix<double, 2> normals(3, 2);
    normals << .6, .8,
               -1., 0.,
               .28, -.96;

    Eigen::MatrixXd q;
    flux.local_flux(values, normals, q);

    // The gradient of a linear field is constant, so its flux is the
    // gradient along each normal
    Eigen::VectorXi actives;
    patch.basis().active_on_element(1, actives);

    const Eigen::Vector2d gradient(1., -2.);
    const Eigen::VectorXd u =
        patch.coefficients()(actives, Eigen::placeholders::all) * gradient;

    REQUIRE((q.transpose() * u - normals * gradient).cwiseAbs().maxCoeff()
            < 1e-14);
}
