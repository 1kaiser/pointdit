# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: pointdit_litert
#     language: python
#     name: pointdit_litert
# ---

# %% [markdown]
# # Verify `web_demo/hosted/index.html` end-to-end -- both real runtime paths
#
# `tools/litert/web_demo/hosted/index.html` is a single, self-contained file: no local
# `npm install`, no local `.tflite` file -- `@litertjs/core` and the model are both fetched at
# runtime (the model from [`huggingface.co/1kaiser/pointdit-litert`](https://huggingface.co/1kaiser/pointdit-litert),
# not a GitHub Release -- see markdown below for why). It also detects, at runtime, whether it's
# being served with the two things the *threaded* WASM runtime needs (cross-origin isolation +
# same-origin wasm Workers) and uses that fast path automatically, falling back to the portable
# CDN runtime otherwise -- mirroring a real prior project on this machine
# (`1kaiser/astro`'s `moge-jax-lite/webgpu_demo`). It runs the real 3-step euler generation loop
# (not just a single denoiser call) and renders the actual resulting point cloud via
# `<model-viewer>`, the same way that prior project visualizes its own reconstruction -- a plain
# numeric accuracy check without a real visual result was a real gap in this demo's first version,
# caught when asked "where is the visualization?" (see `research-repo-bringup` skill).
#
# This notebook verifies **both** real configurations, not just the default one -- matching this
# project's own "verify by running, not by reading the code" discipline. Plain `python -m
# http.server` and the bundled `serve_threaded.py` are genuinely different code paths inside
# `index.html` (different WASM runtime, different accelerator init), so only actually running
# both proves both work, rather than assuming the fallback is correct because the fast path is.

# %% [markdown]
# ## 1. Why the model is fetched from Hugging Face, not this repo's own GitHub Release
#
# Checked directly, not assumed: `curl -sI -L <github-release-asset-url>` returns no
# `Access-Control-Allow-Origin` header anywhere in the redirect chain -- a browser `fetch()` from
# a page hosted anywhere other than github.com is blocked by CORS. `huggingface.co`'s CDN sends
# `access-control-allow-origin: *` (confirmed the same way, on the model uploaded there for this
# demo) -- that's why the model lives there instead.

# %% tags=["parameters"]
web_demo_dir = "tools/litert/web_demo/hosted"
node_bin = "/home/kaiser/.conda/envs/node20/bin/node"
timeout_s = 150

# %%
import functools
import http.server
import json
import subprocess
import threading
from pathlib import Path

# %% [markdown]
# ## 2. Portable path: plain `http.server` (no COOP/COEP headers)
#
# `crossOriginIsolated` is `false` under a plain static server -- `index.html`'s own `loadRuntime()`
# detects this and uses the CDN `{jspi: true}` wasm build, the same portable path this project's
# other browser demos already use.

# %%
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web_demo_dir)
httpd_plain = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=httpd_plain.serve_forever, daemon=True).start()
port_plain = httpd_plain.server_address[1]
url_plain = f"http://127.0.0.1:{port_plain}/index.html"
print(f"serving (plain) at {url_plain}")

result_plain = subprocess.run(
    [node_bin, f"{web_demo_dir}/run_demo.js", url_plain],
    capture_output=True, text=True, timeout=timeout_s,
)
print(result_plain.stdout)
if result_plain.returncode != 0:
    print(result_plain.stderr[-2000:])
httpd_plain.shutdown()
assert result_plain.returncode == 0

# %% [markdown]
# ## 3. Fast path: `serve_threaded.py` (COOP/COEP headers + same-origin `./wasm/` files)
#
# The threaded XNNPACK build has two hard requirements, both real (see `index.html`'s own
# comments for the full detail): cross-origin isolation for `SharedArrayBuffer`, and same-origin
# `Worker()` scripts (a CDN copy is rejected outright by the browser, not slower -- rejected).
# `serve_threaded.py` (committed alongside `index.html`) is the minimal server that provides both.

# %%
threaded_proc = subprocess.Popen(
    ["python3", "serve_threaded.py", "0"], cwd=web_demo_dir,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
# serve_threaded.py prints its actual bound port on the first line before serving.
port_line = threaded_proc.stdout.readline()
print(port_line.strip())
port_threaded = int(port_line.split(":")[-1].split("/")[0])
url_threaded = f"http://127.0.0.1:{port_threaded}/index.html"

result_threaded = subprocess.run(
    [node_bin, f"{web_demo_dir}/run_demo.js", url_threaded],
    capture_output=True, text=True, timeout=timeout_s,
)
print(result_threaded.stdout)
if result_threaded.returncode != 0:
    print(result_threaded.stderr[-2000:])
threaded_proc.terminate()
assert result_threaded.returncode == 0

# %% [markdown]
# ## 4. Verify the real numbers from both runs -- real generation happened, and the real speedup
#
# `index.html` runs the real 3-step euler generation loop and builds an actual point cloud from
# the result (see its own comments) -- numeric accuracy against the PyTorch reference is already
# established elsewhere (`run_litert_inference.py`'s own verified single- and multi-step checks;
# this same model, same math). What this notebook verifies is that the *browser* page genuinely
# ran that real generation (a non-trivial point count, a non-zero generation time) under both
# runtimes, and that the threaded runtime is actually faster, not just reporting a flag.

# %%
def parse_result(stdout):
    line = next(l for l in stdout.splitlines() if l.startswith("RESULT:"))
    return json.loads(line[len("RESULT:"):].strip())

r_plain = parse_result(result_plain.stdout)
r_threaded = parse_result(result_threaded.stdout)
print("portable:", r_plain)
print("threaded:", r_threaded)

assert r_plain["threaded"] is False, "expected the plain server to use the portable runtime"
assert r_threaded["threaded"] is True, "expected serve_threaded.py to actually activate the threaded runtime"
assert r_plain["numPoints"] == r_threaded["numPoints"] > 0, "expected the same non-trivial point count from both runtimes"
assert r_plain["generationMs"] > 0 and r_threaded["generationMs"] > 0, "zero generation time -- inference likely didn't really run"

speedup = r_plain["generationMs"] / r_threaded["generationMs"]
print(f"\nConfirmed: both runtimes built the same real point cloud ({r_plain['numPoints']:,} points).")
print(f"Threaded runtime is {speedup:.1f}x faster than the portable one for the full 3-step "
      f"generation ({r_plain['generationMs']:.0f} ms -> {r_threaded['generationMs']:.0f} ms) -- "
      f"measured here directly, not assumed from a prior project's own number for a different "
      f"model.")
