"""Expose a local dashboard port on the LAN without requiring admin portproxy.

For WSL dashboards that are reachable on Windows localhost, this process binds
to a LAN-facing address and forwards raw TCP streams to the local dashboard.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_handler(target_host: str, target_port: int):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            upstream: socket.socket | None = None
            try:
                upstream = socket.create_connection((target_host, target_port), timeout=10.0)
                sockets = [self.request, upstream]
                for sock in sockets:
                    sock.setblocking(False)
                while True:
                    readable, _, exceptional = select.select(sockets, [], sockets, 60.0)
                    if exceptional or not readable:
                        return
                    for sock in readable:
                        data = sock.recv(65536)
                        if not data:
                            return
                        other = upstream if sock is self.request else self.request
                        other.sendall(data)
            except OSError:
                return
            finally:
                if upstream is not None:
                    upstream.close()
                self.request.close()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8766)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=8765)
    args = parser.parse_args()

    handler = make_handler(args.target_host, args.target_port)
    with ProxyServer((args.listen_host, args.listen_port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
