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
    "software/gui/app.py": "software/gui/app.py",
    "software/gui/hw_cosim.py": "software/gui/hw_cosim.py",
    "software/gui/webgpu_client.py": "software/gui/webgpu_client.py",
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
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/@stlite/browser@0.86.0/build/stlite.css" />
  <style>
    #loading { font-family: sans-serif; padding: 2rem; color: #444; }
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
    files = {}
    for rel, src in FILES.items():
        with open(os.path.join(ROOT, src), "r", encoding="utf-8") as f:
            files[rel] = f.read()
    files_json = json.dumps(files)
    # A literal "</script>" inside embedded sources would close our tag.
    files_json = files_json.replace("</", "<\\/")
    html = (TEMPLATE.replace("__REQUIREMENTS__", json.dumps(REQUIREMENTS))
                    .replace("__FILES__", files_json))
    out = os.path.join(ROOT, "docs", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({len(html)/1e6:.2f} MB, {len(files)} files bundled)")


if __name__ == "__main__":
    main()
