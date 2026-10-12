"""One-command chat: sets the user-local CUDA env, then runs the REPL.

The qdec binary links libcublas from the user-local toolchain (not the
system path), so it dies with "libcublas.so.13: cannot open shared
object file" unless LD_LIBRARY_PATH is set (scripts/cuda_env.sh does
this for shells). This wrapper sets it in-process before spawning, so:
    python3 run_demo_chat.py [--smax 512] [--n 40] [...]

passes everything else through to demo_chat.py. Needs a built binary
in the workdir (it will tell you the exact build command if missing).
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      ".."))
sys.path.insert(0, ROOT)

SP = os.path.join(os.path.expanduser("~"),
                  ".local", "lib", "python3.12", "site-packages")
_CUDA_LIBS = (os.path.join(os.path.expanduser("~"), ".local",
                           "cuda-stubs", "lib"),
              os.path.join(SP, "nvidia", "cu13", "lib"))


def _prepend(key, *dirs):
    have = os.environ.get(key, "")
    ordered = [d for d in list(dirs) + have.split(":")
               if d and os.path.isdir(d)]
    seen, out = set(), []
    for d in ordered:
        if d not in seen:
            seen.add(d)
            out.append(d)
    os.environ[key] = ":".join(out)


def main():
    _prepend("LD_LIBRARY_PATH", *_CUDA_LIBS)
    from demo_chat import main as chat_main
    sys.argv = [os.path.join(ROOT, "demo_chat.py")] + sys.argv[1:]
    chat_main()


if __name__ == "__main__":
    main()
