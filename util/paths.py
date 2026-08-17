# coding=utf-8

"""Paths that resolve against the repository, not the current directory.

Constants the code hardcodes -- ``pretrained/dinov3``, ``generation`` -- go
through :func:`repo_path` instead of being used as bare relative strings, so a
command launched from somewhere other than the repository root still finds the
``pretrained`` and ``datasets`` symlinks that live next to this package.
"""

import os

# The repository root: this file is <repo>/util/paths.py, so go up two levels.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts):
    """Join `parts` onto the repository root and return an absolute path."""
    return os.path.join(REPO_ROOT, *parts)
