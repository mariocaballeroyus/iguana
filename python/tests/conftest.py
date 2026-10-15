# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Configuration of the tests, which skips those that solve on a build
without PETSc"""

import pytest

from iguana import cpp


def pytest_configure(config):
    """Register the marker of the tests that solve."""
    config.addinivalue_line('markers',
                            'petsc: solves, so it needs a build with PETSc')


def pytest_collection_modifyitems(config, items):
    """Skip the tests that solve on a build without PETSc, such as the
    Windows wheel, which has no assembler."""
    if hasattr(cpp, 'Assembler2d'):
        return

    skip = pytest.mark.skip(reason='solving needs a build with PETSc')

    for item in items:
        if 'petsc' in item.keywords:
            item.add_marker(skip)
