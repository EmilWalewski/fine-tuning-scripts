#!/usr/bin/env python3
"""Shared path-profile helpers for portable regeneration manifests."""
import os
import sys

PROFILE_ROOTS = {
    "linux": "/home/atlas/python/traning/fine-tuning-scripts",
    "macos": "/Users/ewalewski/python/fine-tuning-scripts",
    "mac": "/Users/ewalewski/python/fine-tuning-scripts",
    # Container/Docker root used when build_manifest was last run inside the
    # container. Registered as a known SOURCE root so --path-profile auto can
    # remap these paths onto the current host. Not a selectable target profile.
    "container": "/app/fine-tuning-scripts",
}

PROFILE_CHOICES = ("auto", "linux", "macos", "mac", "none")


def host_profile():
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "none"


def profile_root(profile):
    if profile == "auto":
        profile = host_profile()
    if profile == "none":
        return None
    return PROFILE_ROOTS.get(profile)


def map_path(path, profile):
    """Map known project-root prefixes to the requested host profile."""
    if not path:
        return path
    root = profile_root(profile)
    if not root:
        return path

    roots = sorted(set(PROFILE_ROOTS.values()), key=len, reverse=True)
    for known_root in roots:
        if path == known_root or path.startswith(known_root + os.sep):
            rel = os.path.relpath(path, known_root)
            return root if rel == "." else os.path.join(root, rel)
    return path


def mapped_entry(entry, profile):
    out = dict(entry)
    if "input_path" in out:
        out["input_path"] = map_path(out["input_path"], profile)
    if "output_path" in out:
        out["output_path"] = map_path(out["output_path"], profile)
    return out


def add_path_profile_arg(parser, *, verb="use"):
    parser.add_argument(
        "--path-profile",
        choices=PROFILE_CHOICES,
        default="auto",
        help=(
            f"Path profile to {verb}: auto maps manifest paths to this OS "
            "(Linux=/home/atlas/..., macOS=/Users/ewalewski/...), "
            "none leaves paths unchanged."
        ),
    )
