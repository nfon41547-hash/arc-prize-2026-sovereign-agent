"""Vendored arc3sdk runtime — minimal, dependency-free, offline-first.

Shipped inside the starter kernel (see vendor cell) so the worker never depends
on the pinned runtime dataset revision. Only the modules below exist here;
everything else must fail-open at the call site.
"""
__version__ = "v32-vendored"
VENDORED = True
