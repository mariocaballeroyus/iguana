# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Smoke test of the iguana extension module

Importing the module resolves its symbols at load time, which linking
the extension alone does not exercise
"""

import iguana


def test_module_reports_its_version():
    assert isinstance(iguana.__version__, str)
    assert iguana.__version__
