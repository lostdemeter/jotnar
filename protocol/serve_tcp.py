"""Standalone TCP node runner (protocol Milestone 2).

Serves one loopback.Server on a TCP port (one client at a time;
second connection gets a loud busy-refusal). Model-blind: socket +
loopback imports only.
Usage: python3 protocol/serve_tcp.py [--host 127.0.0.1] [--port 0]
  Prints READY <port> on stdout when listening; Ctrl-C stops.
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from protocol.loopback import Server
from protocol.tcp import TcpServer


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--backends", default="lattice,served",
                    help="comma-separated backends to advertise "
                    "(e.g. lattice for a CPU-only node)")
    args = ap.parse_args()
    allowed = [b.strip() for b in args.backends.split(",") if b.strip()]
    srv = TcpServer(Server(allowed=allowed), host=args.host,
                    port=args.port, timeout=args.timeout)
    print(f"READY {srv.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


if __name__ == "__main__":
    main()
