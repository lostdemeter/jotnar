"""C-lowering gate: builds and runs c_chain on this machine.

Proves the bit-exact claim beyond numpy: the C sqrt/mul lowering agrees
with the reference on the shared vector table (see test_holo_c.c and
test_core.py c-vectors-*). Plus the float trap on C sources.
Usage: python3 test_c.py
"""
import os
import subprocess
import sys

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cc = os.path.join(root, "c_chain")
    r = subprocess.run(["make", "-C", cc, "check"], capture_output=True, text=True)
    print(r.stdout[-2000:])
    check("c-check", r.returncode == 0, f"exit={r.returncode}")
    if r.returncode:
        print(r.stderr[-2000:])
    r2 = subprocess.run(["make", "-C", cc, "trap"], capture_output=True, text=True)
    check("c-trap", r2.returncode == 0 and "no float" in r2.stdout, r2.stdout.strip().splitlines()[-1] if r2.stdout else "")
    r3 = subprocess.run([sys.executable, os.path.join(root, "test_c_conv.py")],
                        capture_output=True, text=True, cwd=root)
    print(r3.stdout[-1500:])
    check("c-conv-exchange", r3.returncode == 0, f"exit={r3.returncode}")
    if r3.returncode:
        print(r3.stderr[-1500:])
    r4 = subprocess.run([sys.executable, os.path.join(root, "test_splat_c.py")],
                        capture_output=True, text=True, cwd=root)
    print(r4.stdout[-1500:])
    check("c-splat-exchange", r4.returncode == 0, f"exit={r4.returncode}")
    if r4.returncode:
        print(r4.stderr[-1500:])
    r6 = subprocess.run([sys.executable, os.path.join(root, "test_matmul_c.py")],
                        capture_output=True, text=True, cwd=root)
    print(r6.stdout[-1500:])
    check("c-matmul-exchange", r6.returncode == 0, f"exit={r6.returncode}")
    if r6.returncode:
        print(r6.stderr[-1500:])
    r7 = subprocess.run([sys.executable, os.path.join(root, "test_matmul_cuda.py")],
                        capture_output=True, text=True, cwd=root)
    print(r7.stdout[-1500:])
    check("c-cuda-exchange", r7.returncode == 0, f"exit={r7.returncode}")
    if r7.returncode:
        print(r7.stderr[-1500:])
    r5 = subprocess.run([sys.executable, os.path.join(root, "test_v4.py")],
                        capture_output=True, text=True, cwd=root)
    print(r5.stdout[-1500:])
    check("v4-gates", r5.returncode == 0, f"exit={r5.returncode}")
    if r5.returncode:
        print(r5.stderr[-1500:])
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
