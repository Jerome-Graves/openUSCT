"""Client-GPU acoustic FMC for the in-browser (stlite/Pyodide) build.

Runs the acoustic full-matrix-capture forward model on the *visitor's* GPU via
WebGPU compute shaders, orchestrated from Python through the Pyodide-JS
bridge. WebGPU is asynchronous and the Streamlit script is synchronous, so the
API is job-based: :func:`start` launches the whole FMC on the GPU and returns
immediately; the app polls :func:`poll` across Streamlit reruns (the worker's
event loop advances the GPU promises between reruns) and fetches the data with
:func:`result` when done.

The WGSL step kernel is the exact discrete leapfrog of ringfwi.solver: the
7-point Laplacian on interior cells only (one-cell border stays zero), source
sample added to the Laplacian before the m-scaling, receivers recorded at each
new time level. float32 on the GPU; the app verifies one transmit against the
CPU reference before accepting the data.

On desktop Python there is no ``js`` module, so :func:`available` is False and
nothing here runs.
"""

from __future__ import annotations

import numpy as np

_JS = r"""
globalThis.__OU = globalThis.__OU || { jobs: {}, device: null, pipe: null };

async function __ouDevice() {
  if (globalThis.__OU.device) return globalThis.__OU.device;
  const ad = await navigator.gpu.requestAdapter();
  if (!ad) throw new Error("no WebGPU adapter");
  globalThis.__OU.device = await ad.requestDevice();
  return globalThis.__OU.device;
}

const __OU_WGSL = `
struct U {
  nz: u32, ny: u32, nx: u32, srcLin: u32,
  step: u32, nrx: u32, pad0: u32, pad1: u32,
  wav: f32, invH2: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> P: U;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> pPrev: array<f32>;
@group(0) @binding(3) var<storage, read> pCur: array<f32>;
@group(0) @binding(4) var<storage, read_write> pNew: array<f32>;
@group(0) @binding(5) var<storage, read> recLin: array<u32>;
@group(0) @binding(6) var<storage, read_write> rec: array<f32>;

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let nyx = P.ny * P.nx;
  let N = P.nz * nyx;
  if (i >= N) { return; }
  let iz = i / nyx;
  let rem = i % nyx;
  let iy = rem / P.nx;
  let ix = rem % P.nx;
  var lap: f32 = 0.0;
  // Interior-only stencil: the one-cell border keeps lap = 0 (matches the
  // Python reference).
  if (ix > 0u && ix < P.nx - 1u && iy > 0u && iy < P.ny - 1u
      && iz > 0u && iz < P.nz - 1u) {
    lap = -6.0 * pCur[i]
        + pCur[i - 1u] + pCur[i + 1u]
        + pCur[i - P.nx] + pCur[i + P.nx]
        + pCur[i - nyx] + pCur[i + nyx];
    lap = lap * P.invH2;
  }
  if (i == P.srcLin) { lap = lap + P.wav; }
  pNew[i] = 2.0 * pCur[i] - pPrev[i] + S[i] * lap;
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.nrx) { return; }
  rec[P.step * P.nrx + j] = pNew[recLin[j]];
}
`;

globalThis.__ouStartFmc = function (id, nz, ny, nx, nt, invH2,
                                    sB, wavB, recLinB, srcLinB, nrx) {
  const job = { done: false, prog: 0, total: 0, error: null, result: null };
  globalThis.__OU.jobs[id] = job;
  const asU8 = (b) => (b instanceof Uint8Array) ? b : new Uint8Array(b);
  (async () => {
    try {
      const dev = await __ouDevice();
      const N = nz * ny * nx;
      const S = new Float32Array(asU8(sB).slice().buffer);
      const wav = new Float32Array(asU8(wavB).slice().buffer);
      const recLin = new Uint32Array(asU8(recLinB).slice().buffer);
      const srcLin = new Uint32Array(asU8(srcLinB).slice().buffer);
      const nTx = srcLin.length;
      job.total = nTx;

      const mk = (sz, usage) => dev.createBuffer({ size: sz, usage: usage });
      const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const bufS = mk(N * 4, ST);
      dev.queue.writeBuffer(bufS, 0, S);
      const bufRecLin = mk(Math.max(nrx, 1) * 4, ST);
      dev.queue.writeBuffer(bufRecLin, 0, recLin);
      const p = [mk(N * 4, ST | GPUBufferUsage.COPY_SRC),
                 mk(N * 4, ST | GPUBufferUsage.COPY_SRC),
                 mk(N * 4, ST | GPUBufferUsage.COPY_SRC)];
      const bufRec = mk(nt * nrx * 4, ST | GPUBufferUsage.COPY_SRC);
      const uStride = 256;
      const bufU = dev.createBuffer({
        size: nt * uStride,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const staging = dev.createBuffer({
        size: nt * nrx * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      if (!globalThis.__OU.pipe) {
        const mod = dev.createShaderModule({ code: __OU_WGSL });
        const bgl = dev.createBindGroupLayout({ entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform", hasDynamicOffset: true } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" } },
          { binding: 6, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" } },
        ]});
        const lay = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
        globalThis.__OU.pipe = {
          bgl: bgl,
          step: dev.createComputePipeline({
            layout: lay, compute: { module: mod, entryPoint: "step" } }),
          gather: dev.createComputePipeline({
            layout: lay, compute: { module: mod, entryPoint: "gather" } }),
        };
      }
      const pipe = globalThis.__OU.pipe;

      // Three bind groups rotate the (prev, cur, new) roles of the p buffers.
      const bgs = [];
      for (let r = 0; r < 3; r++) {
        bgs.push(dev.createBindGroup({ layout: pipe.bgl, entries: [
          { binding: 0, resource: { buffer: bufU, size: 48 } },
          { binding: 1, resource: { buffer: bufS } },
          { binding: 2, resource: { buffer: p[r % 3] } },
          { binding: 3, resource: { buffer: p[(r + 1) % 3] } },
          { binding: 4, resource: { buffer: p[(r + 2) % 3] } },
          { binding: 5, resource: { buffer: bufRecLin } },
          { binding: 6, resource: { buffer: bufRec } },
        ]}));
      }

      const result = new Float32Array(nTx * nt * nrx);
      const uData = new ArrayBuffer(nt * uStride);
      for (let t = 0; t < nTx; t++) {
        for (let n = 1; n < nt; n++) {
          const dv = new DataView(uData, n * uStride, 48);
          dv.setUint32(0, nz, true); dv.setUint32(4, ny, true);
          dv.setUint32(8, nx, true); dv.setUint32(12, srcLin[t], true);
          dv.setUint32(16, n, true); dv.setUint32(20, nrx, true);
          dv.setFloat32(32, wav[n - 1], true);
          dv.setFloat32(36, invH2, true);
        }
        dev.queue.writeBuffer(bufU, 0, uData);
        const enc = dev.createCommandEncoder();
        for (const b of p) enc.clearBuffer(b);
        enc.clearBuffer(bufRec);
        const pass = enc.beginComputePass();
        const wgN = Math.ceil(N / 64), wgR = Math.ceil(nrx / 64);
        for (let n = 1; n < nt; n++) {
          const bg = bgs[(n - 1) % 3];
          pass.setPipeline(pipe.step);
          pass.setBindGroup(0, bg, [n * uStride]);
          pass.dispatchWorkgroups(wgN);
          pass.setPipeline(pipe.gather);
          pass.setBindGroup(0, bg, [n * uStride]);
          pass.dispatchWorkgroups(wgR);
        }
        pass.end();
        enc.copyBufferToBuffer(bufRec, 0, staging, 0, nt * nrx * 4);
        dev.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        result.set(new Float32Array(staging.getMappedRange().slice(0)),
                   t * nt * nrx);
        staging.unmap();
        job.prog = t + 1;
        try {
          globalThis.__ouProgChan = globalThis.__ouProgChan
            || new BroadcastChannel('ou_progress');
          globalThis.__ouProgChan.postMessage({
            frac: (t + 1) / nTx,
            text: 'Acquisition on YOUR GPU (WebGPU): transmit '
                  + (t + 1) + '/' + nTx });
        } catch (e) {}
      }
      job.result = result;
      job.done = true;
    } catch (e) {
      job.error = String(e);
      job.done = true;
    }
  })();
};
"""

_loaded = False
_counter = [0]


def available():
    """True inside a browser (Pyodide) whose runtime exposes WebGPU."""
    try:
        import js
        return getattr(js.navigator, "gpu", None) is not None
    except Exception:
        return False


def _ensure_js():
    global _loaded
    if not _loaded:
        import js
        js.eval(_JS)
        _loaded = True


def start(m, geom, wavelet, dt, h, nt, src_list):
    """Launch the full FMC on the client GPU; returns a job id immediately."""
    import js
    from pyodide.ffi import to_js

    _ensure_js()
    m = np.asarray(m, float)
    nz, ny, nx = m.shape
    S = (dt * dt / m).astype(np.float32).ravel()

    def lin(t):
        return (t[0] * ny + t[1]) * nx + t[2]

    rec_lin = np.array([lin(t) for t in geom.idx], dtype=np.uint32)
    src_lin = np.array([lin(geom.element_index(s)) for s in src_list],
                       dtype=np.uint32)
    _counter[0] += 1
    job_id = f"fmc{_counter[0]}"
    js.__ouStartFmc(job_id, nz, ny, nx, int(nt), float(1.0 / (h * h)),
                    to_js(S.tobytes()),
                    to_js(np.asarray(wavelet, np.float32).tobytes()),
                    to_js(rec_lin.tobytes()), to_js(src_lin.tobytes()),
                    int(len(rec_lin)))
    return job_id


def poll(job_id):
    """{'done': bool, 'prog': int, 'total': int, 'error': str|None}."""
    import js
    job = getattr(js.__OU.jobs, job_id)
    return dict(done=bool(job.done), prog=int(job.prog),
                total=int(job.total), error=(str(job.error) if job.error else None))


def result(job_id, n_tx, nt, n_rx):
    """Fetch the finished FMC data as (n_tx, nt, n_rx) float64."""
    import js
    job = getattr(js.__OU.jobs, job_id)
    buf = np.asarray(job.result.to_py(), dtype=np.float32)
    js.eval("delete globalThis.__OU.jobs['" + job_id + "']")
    return buf.reshape(n_tx, nt, n_rx).astype(np.float64)
