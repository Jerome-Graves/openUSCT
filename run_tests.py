"""Master test runner: runs every layer that is available on this machine.

Always runs the pure-Python tests. Runs the C++ parity tests if the extension is
built, the GPU tests if CuPy is installed, and the MATLAB MEX check if MATLAB is
found. Anything unavailable is skipped with a clear reason, so the suite passes
with or without MATLAB (and with or without a C++ toolchain or a GPU).

    python run_tests.py
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
results = []


def _tail(text, n=12):
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n    ".join(lines[-n:])


def run(name, cmd, cwd, env_extra=None, skip_reason=None):
    if skip_reason:
        results.append((name, "SKIP", skip_reason))
        print(f"--- {name}: SKIP ({skip_reason})")
        return
    print(f"--- {name}: running ...")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
        ok = r.returncode == 0
        results.append((name, "PASS" if ok else "FAIL", "" if ok else _tail(r.stdout + r.stderr)))
    except Exception as e:
        results.append((name, "FAIL", str(e)))


def find_matlab():
    m = shutil.which("matlab")
    if m:
        return m
    hits = glob.glob(r"C:\Program Files\MATLAB\*\bin\matlab.exe")
    return hits[0] if hits else None


def main():
    sim = os.path.join(ROOT, "simulation")
    sw = os.path.join(ROOT, "software", "python")

    # 1. Pure-Python core (always).
    run("Python core (pytest)", [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=sim, env_extra={"PYTHONPATH": "."})

    # 2. C++ core parity (if the extension is built).
    uap_built = bool(glob.glob(os.path.join(sw, "_uap*.pyd")) + glob.glob(os.path.join(sw, "_uap*.so")))
    run("C++ core parity", [sys.executable, "test_parity.py"], cwd=sw,
        skip_reason=None if uap_built else "extension not built (software/python: python setup.py build_ext --inplace)")

    # 3. GPU backend (if CuPy is installed).
    try:
        import cupy  # noqa: F401
        has_cupy = True
    except Exception:
        has_cupy = False
    run("GPU backend", [sys.executable, "test_gpu.py"], cwd=sw,
        skip_reason=None if has_cupy else "CuPy not installed")

    # 4. MATLAB MEX (if MATLAB is found).
    matlab = find_matlab()
    mlab_dir = os.path.join(ROOT, "software", "matlab")
    if matlab:
        script = ("if ~isfile('uap_mex.mexw64'); mex('-R2018a','-I../core','uap_mex.cpp'); end; "
                  "uap_gradient_check")
        run("MATLAB MEX", [matlab, "-batch", script], cwd=mlab_dir)
    else:
        run("MATLAB MEX", None, cwd=mlab_dir, skip_reason="MATLAB not found")

    # Summary.
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    width = max(len(n) for n, _, _ in results)
    n_fail = 0
    for name, status, detail in results:
        print(f"  {name:<{width}}  {status}")
        if status == "FAIL":
            n_fail += 1
            print(f"    {detail}")
    print("=" * 60)
    ran = [r for r in results if r[1] != "SKIP"]
    print(f"  {sum(1 for r in ran if r[1]=='PASS')}/{len(ran)} ran and passed; "
          f"{sum(1 for r in results if r[1]=='SKIP')} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
