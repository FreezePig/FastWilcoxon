"""Delocate a wheel, then share GraphBLAS's OpenMP runtime on macOS/Python 3.10+."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from delocate.tools import get_install_names, set_install_name
from delocate.wheeltools import InWheelCtx


# The macOS runtime dependency is pinned in pyproject.toml. Both extensions must
# resolve to this same file, regardless of which package is imported first.
OPENMP_PROVIDER = Path("suitesparse_graphblas/.dylibs/libomp.dylib")


def repair_wheel(wheel, destination, architectures):
    wheel = Path(wheel).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fastwilcoxon_delocate_") as temporary:
        subprocess.run(
            ["delocate-wheel", "--require-archs", architectures,
             "-w", temporary, "-v", str(wheel)],
            check=True,
        )
        repaired, = Path(temporary).glob("*.whl")
        output = destination / repaired.name
        # The Python GraphBLAS backend requires Python >= 3.10. Older Python
        # wheels keep their private OpenMP runtime and existing functionality.
        if sys.version_info < (3, 10):
            shutil.copy2(repaired, output)
            return

        with InWheelCtx(str(repaired)) as context:
            root = Path(context.wheel_path)
            private_runtimes = set(root.rglob("libomp*.dylib"))
            rewritten = 0
            for binary in root.rglob("*"):
                if binary.suffix not in (".so", ".dylib") or binary in private_runtimes:
                    continue
                for dependency in get_install_names(str(binary)):
                    if not Path(dependency).match("libomp*.dylib"):
                        continue
                    relative = os.path.relpath(root / OPENMP_PROVIDER, binary.parent)
                    replacement = "@loader_path/" + Path(relative).as_posix()
                    # delocate also regenerates the Mach-O ad-hoc signature.
                    set_install_name(str(binary), dependency, replacement)
                    print("Shared OpenMP:", binary.relative_to(root), "->", replacement)
                    rewritten += 1
            if not rewritten:
                raise RuntimeError("No OpenMP dependencies found in the macOS wheel")
            for runtime in private_runtimes:
                runtime.unlink()
            # Assign only after a successful edit. InWheelCtx regenerates RECORD
            # and repacks the result when leaving the context.
            context.out_wheel = str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    parser.add_argument("destination")
    parser.add_argument("architectures")
    arguments = parser.parse_args()
    repair_wheel(arguments.wheel, arguments.destination, arguments.architectures)
