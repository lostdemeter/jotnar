"""Pipeline controller (protocol Milestone 3): routes EXECs, ferries boundaries.

One client object per node (one-client-per-server limit stands --
fan-out lives here, not in the servers). Dispatch rule (spec'd in
docs/PROTOCOL.md): served programs go to a node advertising
"served", lattice programs to one advertising "lattice"; no
advertisement, no route (loud). Every routing decision is logged
(node, program, backend) so gates can prove placement.
"""
import sys
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from protocol.loopback import Client  # noqa: F401 (re-exported surface)


class RouteError(RuntimeError):
    """No node advertises the needed backend. Loud, never misrouted."""


class PipeController:
    """Routes programs to nodes by advertised backend capability."""

    def __init__(self):
        self.nodes = {}   # name -> Client
        self.caps = {}    # name -> [backends] from PING
        self.routes = {}  # program -> (node, backend)
        self.log = []     # (program, node, backend) placement record

    def add_node(self, name, client):
        h, _ = client.ping()
        if not h.get("ok"):
            raise RouteError(f"node {name} ping failed: {h.get('error')}")
        self.nodes[name] = client
        self.caps[name] = list(h["pong"].get("backends", []))
        return self.caps[name]

    def route_for(self, backend):
        for name, caps in self.nodes.items():
            if backend in caps:
                return name
        raise RouteError(
            f"no node advertises backend {backend!r} "
            f"(have {self.caps})")

    def program(self, node, name, listing, backend="lattice",
                config=None, stamp=None):
        if backend not in self.caps.get(node, []):
            raise RouteError(
                f"node {node} does not advertise {backend!r}")
        h, _ = self.nodes[node].program(name, listing, backend,
                                        config, stamp)
        if not h.get("ok"):
            raise RouteError(f"node {node} refused PROGRAM {name}: "
                             f"{h.get('error')}")
        self.routes[name] = (node, backend)
        self.log.append((name, node, backend))
        return h

    def program_auto(self, name, listing, backend="lattice",
                     config=None, stamp=None):
        return self.program(self.route_for(backend), name, listing,
                            backend, config, stamp)

    def exec(self, name, inputs, outputs):
        node, _ = self.routes[name]
        return self.nodes[node].exec(name, inputs, outputs)

    def placement(self, name):
        return self.routes.get(name)
