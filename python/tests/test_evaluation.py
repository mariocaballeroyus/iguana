# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the fields read back at points"""

import itertools

import numpy as np
import pytest

import iguana
from iguana import (CellType, FunctionSpace, HierarchicalGrid,
                    LinearElasticityElement, MixedFunctionSpace,
                    NitscheCondition, PoissonElement, QuadratureBoundary,
                    QuadratureDomain, StokesElement, SurrogateBoundary,
                    flux_at_points, trace_at_points)

try:
    from iguana.petsc import LinearProblem
except ImportError:
    # A build without PETSc skips every test, as it has no traces
    LinearProblem = None

pytestmark = pytest.mark.petsc

VISCOSITY = 1.5
REFERENCE = (1., 1.6)


def rectangle():
    """Rectangle of sides 2 and 3 with quadratic elements, three along x and
    two along y."""
    return iguana.create_rectangle(lengths=(2., 3.), elements=(3, 2))


def sample_points():
    """Points of the rectangle inside its cells, on its knot lines, along its
    edges and at its corners."""
    return np.array(list(itertools.product(
        [0., .4, 2. / 3., 1.1, 4. / 3., 2.], [0., .7, 1.5, 2.2, 3.])))


def sample_normals(points):
    """Unit normals that differ from point to point."""
    angles = np.linspace(0., 2. * np.pi, len(points), endpoint=False)

    return np.column_stack([np.cos(angles), np.sin(angles)])


def velocity(points):
    """The flow u = (x^2, −2 x y), which has no divergence."""
    x, y = points.T

    return np.column_stack([x * x, -2. * x * y])


def body_force(points):
    """The force f = −div(2μ ε(u)) + ∇p = (1 − 2μ, 1) of the flow, with the
    pressure x + y up to a constant."""
    return np.tile([1. - 2. * VISCOSITY, 1.], (len(points), 1))


def solve_flow(element, space, num_points):
    """Coefficients of the flow on a space, which holds it exactly, imposed
    all around by Nitsche's method, as the problem returns them."""
    patch = space.patch
    quadrature = QuadratureDomain(patch)
    quadrature.fill_gauss_legendre(CellType.inside, num_points)

    num_cells = HierarchicalGrid(patch.degrees, patch.knots).num_elements
    surrogate = SurrogateBoundary(patch, [CellType.inside] * num_cells)
    condition = NitscheCondition(element.u,
                                 QuadratureBoundary(patch, surrogate, 3),
                                 velocity, 40. * element.viscosity)

    return LinearProblem(element, space, quadrature, body_force,
                         conditions=[condition], pressure_reference=REFERENCE,
                         petsc_options_prefix='evaluation_').solve()


def one_field(patch):
    """A stabilized fluid on one field of three components, with the number
    of Gauss points its products need."""
    return StokesElement(VISCOSITY, 3.), FunctionSpace(patch, block_size=3), 3


def a_field_each(patch):
    """A Galerkin fluid on a divergence-conforming space of a field per
    unknown, with the number of Gauss points its products need."""
    return (StokesElement(VISCOSITY),
            MixedFunctionSpace.divergence_conforming(patch), 4)


@pytest.mark.parametrize('setup', [one_field, a_field_each])
def test_flow_reads_back_exactly(setup):
    """A solved flow gives its velocity and its traction, and its pressure
    with the constant of the reference, whose function has zero pressure."""
    patch = rectangle()
    element, space, num_points = setup(patch)
    coefficients = solve_flow(element, space, num_points)

    points = sample_points()
    normals = sample_normals(points)
    x, y = points.T
    n_x, n_y = normals.T

    nearest = np.argmin(np.linalg.norm(patch.control_points - REFERENCE,
                                       axis=1))
    pressure = x + y - patch.control_points[nearest].sum()

    # t = μ (∇u + ∇u^T) n − p n, with ∇u + ∇u^T = [[4x, −2y], [−2y, −4x]]
    traction = (VISCOSITY * np.column_stack([4. * x * n_x - 2. * y * n_y,
                                             -2. * y * n_x - 4. * x * n_y])
                - pressure[:, None] * normals)

    computed = trace_at_points(element.u, space, coefficients, points)
    assert computed.shape == (len(points), 2)
    assert np.allclose(computed, velocity(points), rtol=0., atol=1e-10)

    computed = trace_at_points(element.p, space, coefficients, points)
    assert computed.shape == (len(points),)
    assert np.allclose(computed, pressure, rtol=0., atol=1e-10)

    computed = flux_at_points(element.u, space, coefficients, points, normals)
    assert computed.shape == (len(points), 2)
    assert np.allclose(computed, traction, rtol=0., atol=1e-10)


def test_poisson_field_reads_back():
    """A scalar field gives one value per point, and its normal derivative."""
    patch = rectangle()
    element = PoissonElement()
    space = FunctionSpace(patch)
    slope = np.array([1., -2.])

    # A linear field has its values at the control points as coefficients
    coefficients = patch.control_points @ slope
    points = sample_points()
    normals = sample_normals(points)

    computed = trace_at_points(element.u, space, coefficients, points)
    assert computed.shape == (len(points),)
    assert np.allclose(computed, points @ slope, rtol=0., atol=1e-12)

    computed = flux_at_points(element.u, space, coefficients, points, normals)
    assert computed.shape == (len(points),)
    assert np.allclose(computed, normals @ slope, rtol=0., atol=1e-12)


def test_solid_reads_back():
    """A solid gives its displacement, and its traction from its moduli."""
    patch = rectangle()
    solid = LinearElasticityElement(10., .3)
    space = FunctionSpace(patch, block_size=2)
    gradient = np.array([[1., -2.], [.5, 3.]])

    # A linear field has its values at the control points as coefficients
    coefficients = patch.control_points @ gradient.T
    points = sample_points()
    normals = sample_normals(points)

    # σ = μ (∇u + ∇u^T) + λ (div u) I
    stress = (solid.shear_modulus * (gradient + gradient.T)
              + solid.lame * np.trace(gradient) * np.eye(2))

    computed = trace_at_points(solid.u, space, coefficients, points)
    assert np.allclose(computed, points @ gradient.T, rtol=0., atol=1e-12)

    computed = flux_at_points(solid.u, space, coefficients, points, normals)
    assert np.allclose(computed, normals @ stress, rtol=0., atol=1e-12)


def test_invalid_arguments():
    patch = rectangle()
    fluid = StokesElement(VISCOSITY, 3.)
    space = FunctionSpace(patch, block_size=3)
    coefficients = np.zeros((space.num_dofs, 3))
    points = sample_points()
    normals = sample_normals(points)

    with pytest.raises(TypeError, match='one of an element'):
        trace_at_points(fluid, space, coefficients, points)

    with pytest.raises(TypeError, match='has no flux'):
        flux_at_points(fluid.p, space, coefficients, points, normals)

    with pytest.raises(TypeError, match='function space'):
        trace_at_points(fluid.u, patch, coefficients, points)

    box = iguana.create_box(lengths=(1., 1., 1.), elements=(1, 1, 1),
                            degrees=(1, 1, 1))

    with pytest.raises(TypeError, match='planar patch'):
        trace_at_points(fluid.u, FunctionSpace(box, block_size=3),
                        np.zeros((8, 3)), points)

    with pytest.raises(ValueError, match='each unknown of the element'):
        trace_at_points(PoissonElement().u, space, coefficients, points)

    # Two fields of the sizes of two components, which the rows would not tell
    # apart
    solid = LinearElasticityElement(10., .3)
    mixed = MixedFunctionSpace([FunctionSpace(patch), FunctionSpace(patch)])

    with pytest.raises(ValueError, match='a solid'):
        trace_at_points(solid.u, mixed, np.zeros((2, patch.control_points
                                                  .shape[0])), points)

    with pytest.raises(ValueError, match='every degree of freedom'):
        trace_at_points(fluid.u, space, coefficients[:-1], points)

    with pytest.raises(ValueError, match='every degree of freedom'):
        trace_at_points(fluid.u,
                        MixedFunctionSpace.divergence_conforming(patch),
                        (np.zeros(24), np.zeros(25)), points)

    with pytest.raises(ValueError, match='points of the plane'):
        trace_at_points(fluid.u, space, coefficients, [1., 1.])

    with pytest.raises(ValueError, match='points of the plane'):
        trace_at_points(fluid.u, space, coefficients, [[1., 1., 1.]])

    with pytest.raises(ValueError, match='one per point'):
        flux_at_points(fluid.u, space, coefficients, points, normals[:, 0])

    with pytest.raises(ValueError, match='lie in the patch'):
        trace_at_points(fluid.u, space, coefficients, [[2.5, 1.]])

    # The first cell alone supports the first function
    num_cells = HierarchicalGrid(patch.degrees, patch.knots).num_elements
    kept = FunctionSpace(patch, [CellType.outside]
                         + [CellType.inside] * (num_cells - 1), block_size=3)

    with pytest.raises(ValueError, match='outside the physical domain'):
        trace_at_points(fluid.u, kept, np.zeros((kept.num_dofs, 3)),
                        [[.1, .1]])
