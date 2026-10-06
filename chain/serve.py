"""Persistent-binary driver: weights-load-once, steps in-process.

The loop binaries emitted with live=[...] (chain/emit_cuda.compile_cuda)
read frozen inputs once, then serve steps: host rewrites the live files,
sends a line on stdin, waits for READY on stdout, reads the outputs.
Protocol failures (crash, EOF, timeout, missing READY) raise loudly
with the stderr tail -- never silent wrong bytes.

Timeouts: first READY covers the full weight load (default 900s);
steps default 600s. Fixed-shape contract: live files keep their byte
size across steps (re-malloc never happens in-loop).
"""
import os
import subprocess


class ServeError(RuntimeError):
    """Persistent binary misbehaved (crash/EOF/timeout/protocol)."""


class ServedExe:
    def __init__(self, exe, argv_files, err_path=None):
        self.exe = exe
        self.argv_files = list(argv_files)
        self.err_path = err_path
        self.proc = None
        self._errf = None

    def start(self, timeout=900):
        """Spawn and wait for the first READY (post-load, post-step-0)."""
        if self.err_path:
            self._errf = open(self.err_path, "wb")
        self.proc = subprocess.Popen(
            [self.exe] + self.argv_files, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._errf)
        self._await("startup", timeout)

    def step(self, timeout=600):
        """Host has rewritten the live files: run one step, wait READY."""
        self._check()
        try:
            self.proc.stdin.write(b"step\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise ServeError(self._death("stdin went away on step"))
        self._await("step", timeout)

    def close(self):
        """Ask politely, then wait. Idempotent-ish (never raises)."""
        p = self.proc
        self.proc = None
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.write(b"q\n")
            p.stdin.flush()
            p.wait(timeout=60)
        except Exception:
            p.kill()

    def _check(self):
        if self.proc is None or self.proc.poll() is not None:
            raise ServeError(self._death("binary not running"))

    def _await(self, what, timeout):
        from threading import Timer
        p = self.proc
        line = [None]

        def _kill():
            if p.poll() is None:
                p.kill()

        t = Timer(timeout, _kill)
        t.start()
        try:
            line[0] = p.stdout.readline()
        finally:
            t.cancel()
        if not line[0]:
            raise ServeError(self._death(f"EOF waiting READY ({what})"))
        if line[0].strip() != b"READY":
            raise ServeError(self._death(
                f"protocol: expected READY ({what}), got {line[0][:80]!r}"))

    def _death(self, why):
        tail = ""
        try:
            if self.err_path and os.path.isfile(self.err_path):
                with open(self.err_path, "rb") as f:
                    f.seek(max(0, os.path.getsize(self.err_path) - 600))
                    tail = f.read().decode("utf-8", "replace")[-600:]
        except OSError:
            pass
        rc = self.proc.poll() if self.proc is not None else None
        return f"served exe: {why} (rc={rc}, exe={self.exe}): {tail[:300]}"
