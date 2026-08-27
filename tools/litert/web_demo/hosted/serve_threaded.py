#!/usr/bin/env python3
"""Serve this demo with the fast threaded-WASM path enabled.

Plain `python3 -m http.server` also works for this demo -- index.html detects at runtime
whether it's running cross-origin-isolated and falls back to the portable (CDN wasm,
single-threaded) runtime automatically if not. This script only adds the two things the
faster threaded XNNPACK build needs (see index.html's loadRuntime() for the real, verified
reasons): COOP/COEP response headers, and same-origin wasm files (already committed under
./wasm/ in this directory -- the threaded runtime spawns real Worker()s, which reject
cross-origin/CDN script URLs).

Usage: python3 serve_threaded.py [port]   (default 8791)
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # SharedArrayBuffer (threaded WASM) requires cross-origin isolation.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


requested_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
server = ThreadingHTTPServer(("127.0.0.1", requested_port), Handler)
# Print the real bound port, not the requested one -- requesting port 0 asks the OS for any free
# port, and server_address only reflects the real choice after the socket is actually bound.
# flush=True: stdout is fully (block-)buffered rather than line-buffered when it's not a TTY (e.g.
# a subprocess.PIPE), so a caller reading this line to learn the real port can otherwise block
# forever waiting for output that's sitting in an internal buffer this process never flushes on
# its own (serve_forever() produces no further output to force a flush).
print(f"serving on http://localhost:{server.server_address[1]}/index.html (threaded fast path enabled)", flush=True)
server.serve_forever()
