"""Locate build-only oneMKL archives inside a PEP 517 isolated environment."""

from importlib.metadata import distribution
from pathlib import Path


def installed_file(package, basename):
    dist = distribution(package)
    paths = [Path(dist.locate_file(p)).resolve()
             for p in (dist.files or ()) if p.name == basename]
    if len(paths) != 1 or not paths[0].is_file():
        raise RuntimeError(f"{package}: expected one installed {basename}, found {paths}")
    return paths[0]


if __name__ == "__main__":
    paths = [
        installed_file("mkl-include", "mkl.h").parent,
        installed_file("mkl-static", "libmkl_intel_ilp64.a"),
        installed_file("mkl-static", "libmkl_gnu_thread.a"),
        installed_file("mkl-static", "libmkl_core.a"),
        installed_file("mkl-static", "LICENSE.txt"),
    ]
    print(";".join(path.as_posix() for path in paths))
