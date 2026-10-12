"""TCP transport for the v1 wire protocol (Milestone 2).

Same frames as protocol/loopback.py (u32BE length + JSON + payload) --
the codec is reused byte-for-byte, not reinvented. Policy, explicit:
ONE client at a time, single-threaded (select over listener + active
connection). A second connection while one is active gets an
`{ok:false, code:"busy"}` frame and a close -- loud, never a queue.
Every socket carries a timeout on both ends (connect + recv), so a
dropped peer raises instead of hanging. No TLS/auth (deferred, and
the frames state it nowhere -- plain TCP, documented).

Server/transport code stays model-blind (gate-grepped like the
loopback SERVER section): socket + loopback imports only.
"""
import select
import socket
import struct

from protocol.loopback import Server, Client, dumps, loads, FrameError


class TransportError(RuntimeError):
    """Dropped/unreachable peer (timeout, reset, refused). Loud, no hangs."""


def send_frame(sock, frame):
    try:
        sock.sendall(frame)
    except (socket.timeout, ConnectionResetError, BrokenPipeError) as e:
        raise TransportError(f"send failed: {e}")


def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout as e:
            raise TransportError(f"recv timeout: {e}")
        if not chunk:
            raise TransportError("peer closed mid-frame")
        buf += chunk
    return buf


def recv_frame(sock):
    try:
        raw = _recvn(sock, 4)
    except TransportError:
        raise
    (n,) = struct.unpack(">I", raw)
    if n > 256 * 1024 * 1024:
        raise TransportError(f"frame too big: {n}")
    return raw + _recvn(sock, n)


class TcpServer:
    """One-client-at-a-time socket front for a loopback.Server."""

    def __init__(self, server=None, host="127.0.0.1", port=0, timeout=600,
                 backends=None):
        self.inner = server if server is not None else Server(allowed=backends)
        self.timeout = timeout
        self._stop = False
        self.lsn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsn.bind((host, port))
        self.lsn.listen(1)
        self.port = self.lsn.getsockname()[1]

    def stop(self):
        self._stop = True

    def close(self):
        try:
            self.lsn.close()
        except OSError:
            pass

    def _busy_reply(self, conn):
        try:
            send_frame(conn, dumps("error", {"id": 0, "ok": False,
                                             "error": {"code": "busy",
                                                       "msg": "one client at a time"}}))
        except TransportError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def serve_forever(self):
        active = None
        while not self._stop:
            want = [self.lsn] if active is None else [self.lsn, active]
            try:
                r, _, _ = select.select(want, [], [], 1.0)
            except (OSError, ValueError):
                break
            for s in r:
                if s is self.lsn:
                    try:
                        conn, _ = self.lsn.accept()
                    except OSError:
                        continue
                    if active is not None:
                        self._busy_reply(conn)
                    else:
                        try:
                            conn.settimeout(self.timeout)
                        except OSError:
                            conn.close()
                        else:
                            active = conn
                else:
                    try:
                        frame = recv_frame(active)
                    except TransportError:
                        try:
                            active.close()
                        except OSError:
                            pass
                        active = None
                        continue
                    try:
                        resp = self.inner.handle(frame)
                    except Exception as e:  # never kill the connection
                        resp = dumps("error", {"id": 0, "ok": False,
                                               "error": {"code": "node",
                                                         "msg": str(e)[:300]}})
                    try:
                        send_frame(active, resp)
                    except TransportError:
                        try:
                            active.close()
                        except OSError:
                            pass
                        active = None
        if active is not None:
            try:
                active.close()
            except OSError:
                pass


class TcpClient(Client):
    """Same call surface as the loopback client, socket-backed."""

    def __init__(self, host="127.0.0.1", port=0, timeout=600):
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            raise TransportError(f"connect {host}:{port} failed: {e}")
        try:
            sock.settimeout(timeout)
        except OSError as e:
            sock.close()
            raise TransportError(f"settimeout failed: {e}")
        self.sock = sock
        super().__init__(transport=self._transport)

    def _transport(self, frame):
        send_frame(self.sock, frame)
        try:
            return recv_frame(self.sock)
        except TransportError as e:
            raise TransportError(f"roundtrip failed: {e}")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
