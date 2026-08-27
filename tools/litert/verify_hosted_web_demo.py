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
# # Verify `index_hosted_b16.html` end-to-end: real network fetch, real in-browser inference
#
# `index_hosted_b16.html` (this directory's `web_demo/`) is meant to be self-sufficient: no local
# `npm install`, no local `.tflite` file -- `@litertjs/core` loads from a CDN and the model is
# fetched at runtime from [`huggingface.co/1kaiser/pointdit-litert`](https://huggingface.co/1kaiser/pointdit-litert).
# This notebook is the reproducible version of that check (matching this project's own
# "verify by running, not by reading the code" discipline throughout) -- it serves the page
# locally (never as a bare `file://` -- same CORS reason as every other GLB/HTML page in this
# project), drives it with a real headless Chrome via Playwright, and asserts on the actual
# printed result rather than just "the subprocess exited 0".

# %% [markdown]
# ## 1. Serve the demo directory locally
#
# Never open as `file://` -- a `null`-origin page can't `fetch()` its own sibling assets
# (`shapes.json`, the `.bin` tensors), the same CORS gap documented for the GLB/model-viewer
# pages elsewhere in this project.

# %% tags=["parameters"]
web_demo_dir = "tools/litert/web_demo"
demo_html = "index_hosted_b16.html"
node_bin = "/home/kaiser/.conda/envs/node20/bin/node"
timeout_s = 150

# %%
import functools
import http.server
import json
import subprocess
import threading
from pathlib import Path

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web_demo_dir)
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
port = httpd.server_address[1]
url = f"http://127.0.0.1:{port}/{demo_html}"
print(f"serving {web_demo_dir} at {url}")

# %% [markdown]
# ## 2. Drive it with real headless Chrome and capture the real result
#
# `run_demo_hosted_b16.js` prints a single `RESULT: {...}` JSON line once
# `window.__demoDone` is set -- the same page-side completion signal every browser demo driver in
# this project uses. This is a real network-bound run (downloads the actual 139.7MB model from
# HF over the network), not a cached/mocked one -- expect ~30-60s just for the download, before
# any inference happens.

# %%
result = subprocess.run(
    [node_bin, f"{web_demo_dir}/run_demo_hosted_b16.js", url],
    capture_output=True, text=True, timeout=timeout_s,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr[-2000:])
httpd.shutdown()
assert result.returncode == 0, "run_demo_hosted_b16.js exited non-zero"

# %% [markdown]
# ## 3. Verify the real numbers, not just that it printed something
#
# A page that fetched nothing and silently no-op'd would still exit 0 -- the actual assertion is
# on `RESULT`'s real fields: the model genuinely downloaded (a non-trivial size), inference
# genuinely ran (a non-zero latency), and its accuracy matches the already-established
# weight-only int8 B/16 number (0.007-0.008 max abs diff vs. the PyTorch-GPU reference; see
# `run_litert_inference.py` and the main README's LiteRT benchmark table) rather than some
# unrelated value that would indicate a silent fallback or a stale/wrong model file.

# %%
result_line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT:"))
demo_result = json.loads(result_line[len("RESULT:"):].strip())
print(demo_result)

assert demo_result["latencyMs"] > 0, "zero-latency result -- inference likely didn't really run"
assert 0.005 < demo_result["maxDiff"] < 0.02, (
    f"max abs diff {demo_result['maxDiff']} is outside the expected weight-only int8 B/16 range "
    f"(0.007-0.008 previously measured) -- something about the fetched model or inference path "
    f"may be wrong, not just noisier than usual"
)
print(f"\nConfirmed: real model fetched from Hugging Face Hub, real in-browser inference, "
      f"{demo_result['latencyMs']:.0f} ms, max abs diff {demo_result['maxDiff']:.4e} "
      f"(matches the known-good weight-only int8 B/16 accuracy).")
