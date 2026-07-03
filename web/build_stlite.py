"""Build a self-contained browser version of OpenUSCT Studio (stlite).

Bundles the Streamlit app and the ringfwi package into a single static
``docs/index.html`` that runs entirely in the visitor's browser via stlite
(Streamlit on Pyodide/WebAssembly): all computation happens on the client's
machine, no Python server or container involved. Host it with GitHub Pages
(repo Settings -> Pages -> deploy from branch, folder /docs).

Run:  python web/build_stlite.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    ".streamlit/config.toml": ".streamlit/config.toml",
    "software/gui/app.py": "software/gui/app.py",
    "software/gui/hw_cosim.py": "software/gui/hw_cosim.py",
    "software/gui/webgpu_client.py": "software/gui/webgpu_client.py",
    "software/gui/webgpu_elastic.py": "software/gui/webgpu_elastic.py",
}
RINGFWI = os.path.join(ROOT, "simulation", "ringfwi")
for fn in sorted(os.listdir(RINGFWI)):
    if fn.endswith(".py"):
        FILES[f"simulation/ringfwi/{fn}"] = f"simulation/ringfwi/{fn}"

REQUIREMENTS = ["numpy", "scipy", "matplotlib", "plotly", "scikit-image", "h5py"]

TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <title>OpenUSCT Studio (in-browser)</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg" />
  <link rel="alternate icon" href="favicon.ico" />
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/@stlite/browser@0.86.0/build/stlite.css" />
  <style>
    body { background: #0a1017; }
    #loading { font-family: sans-serif; padding: 2rem; color: #9fb8cc; }
    #loading h2 { color: #7fcbe8; }
  </style>
</head>
<body>
  <div id="root">
    <div id="loading">
      <h2>OpenUSCT Studio</h2>
      <p>Loading the in-browser Python runtime (first visit downloads
         ~60&nbsp;MB of scientific packages; they are cached afterwards).</p>
      <p>All computation runs on <b>your</b> machine &mdash; nothing is sent to a
         server.</p>
    </div>
  </div>
  <script>
  // Capture stlite's worker(s): the worker's event loop dispatches DIRECT
  // postMessage events (its own kernel traffic proves it) but not
  // BroadcastChannel messages, so GPU results must be delivered by
  // worker.postMessage.
  (function () {
    window.__ouWorkers = [];
    const OrigWorker = window.Worker;
    window.Worker = function (...args) {
      const w = new OrigWorker(...args);
      window.__ouWorkers.push(w);
      return w;
    };
    window.Worker.prototype = OrigWorker.prototype;
  })();
  // Main-thread GPU engines + job bridge: stlite parks its Python worker so
  // WebGPU promises never advance there; jobs posted on the 'ou_gpu'
  // BroadcastChannel run HERE (where the GPU works) and results are
  // delivered back into the worker via postMessage.
__GPU_ENGINES__
  (function () {
    const ch = new BroadcastChannel('ou_gpu');
    function toWorkers(msg, transfer) {
      for (const w of (window.__ouWorkers || [])) {
        try { w.postMessage(msg, transfer || []); } catch (e) {}
      }
    }
    function watch(id, job) {
      const iv = setInterval(() => {
        if (job.done) {
          clearInterval(iv);
          const out = { __ougpu: true, type: 'done', id: id,
                        error: job.error || null };
          if (!job.error && job.result) {
            out.result = job.result.buffer;
            toWorkers(out, [out.result]);
          } else {
            toWorkers(out);
          }
        } else {
          toWorkers({ __ougpu: true, type: 'prog', id: id,
                      prog: job.prog, total: job.total });
        }
      }, 250);
    }
    ch.onmessage = (ev) => {
      const m = ev.data || {};
      try {
        if (m.type === 'el_start') {
          __oueStart(m.id, m.nz, m.ny, m.nx, m.B, m.nt, m.invh, m.dt,
                     m.mat, m.wav, m.recLin, m.src, m.nrx, m.nsrc);
          watch(m.id, globalThis.__OUE.jobs[m.id]);
        } else if (m.type === 'ac_start') {
          __ouStartFmc(m.id, m.nz, m.ny, m.nx, m.nt, m.invH2,
                       m.S, m.wav, m.recLin, m.srcLin, m.nrx);
          watch(m.id, globalThis.__OU.jobs[m.id]);
        }
      } catch (e) {
        if (m.id) { ch.postMessage({ type: 'done', id: m.id,
                                     error: String(e) }); }
      }
    };
  })();
  </script>
  <script type="module">
    import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@0.86.0/build/stlite.js";
    mount(
      {
        requirements: __REQUIREMENTS__,
        entrypoint: "software/gui/app.py",
        files: __FILES__,
      },
      document.getElementById("root"),
    );
  </script>
</body>
</html>
"""


def main():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "software", "gui"))
    import webgpu_client
    import webgpu_elastic
    engines = (webgpu_elastic._js_source() + "\n"
               + webgpu_client._js_source())
    assert "</" not in engines, "script-closing sequence in engine JS"

    files = {}
    for rel, src in FILES.items():
        with open(os.path.join(ROOT, src), "r", encoding="utf-8") as f:
            files[rel] = f.read()
    files_json = json.dumps(files)
    # A literal "</script>" inside embedded sources would close our tag.
    files_json = files_json.replace("</", "<\\/")
    html = (TEMPLATE.replace("__REQUIREMENTS__", json.dumps(REQUIREMENTS))
                    .replace("__GPU_ENGINES__", engines)
                    .replace("__FILES__", files_json))
    out = os.path.join(ROOT, "docs", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({len(html)/1e6:.2f} MB, {len(files)} files bundled)")


if __name__ == "__main__":
    main()
