"""Client-GPU 3D anisotropic ELASTIC forward for the in-browser build.

Runs the full 21-component staggered-grid elastic solver (the physics of
ringfwi.elastic3d.forward) on the visitor's GPU via WebGPU compute shaders,
batched over transmits -- the browser counterpart of the batched CuPy path,
so the Voronoi-seed inversion and polycrystal acquisition reach native-like
speed online.

The WGSL stress-update terms are GENERATED from the same staggered-position
tables (_POS/_VPOS) the verified CPU solver uses, so the shader cannot drift
from the reference physics. float32 on the GPU; callers verify the first
evaluation against the CPU reference before trusting the data (same
self-check-then-fallback contract as the acoustic WebGPU path).

Job-based API like webgpu_client: :func:`start` returns immediately,
:func:`poll` across Streamlit reruns, :func:`result` fetches (B, nt, n_rx).
"""

from __future__ import annotations

import numpy as np

KEYS21 = tuple(f"C{i}{j}" for i in range(1, 7) for j in range(i, 7))
_POS = {"c": (0, 0, 0), "yz": (1, 1, 0), "xz": (1, 0, 1), "xy": (0, 1, 1)}
_VPOS = {1: "c", 2: "c", 3: "c", 4: "yz", 5: "xz", 6: "xy"}


def _gen_stress_terms():
    """WGSL statements for s_I += dt * sum_J C_IJ * move(gam_J, J->I)."""
    lines = []
    for I in range(1, 7):
        posI = _POS[_VPOS[I]]
        lines.append("  { var acc: f32 = 0.0;")
        for J in range(1, 7):
            key = f"C{min(I, J)}{max(I, J)}"
            kidx = KEYS21.index(key)
            posJ = _POS[_VPOS[J]]
            dz, dy, dx = (posI[0] - posJ[0], posI[1] - posJ[1],
                          posI[2] - posJ[2])
            lines.append(
                f"    acc = acc + MAT[{kidx}u * P.N + cell] "
                f"* mvg({J - 1}u, b, iz, iy, ix, {dz}, {dy}, {dx});")
        lines.append(
            f"    F[({2 + I}u) * P.BN + b * P.N + cell] = "
            f"F[({2 + I}u) * P.BN + b * P.N + cell] + P.dt * acc; }}")
    return "\n".join(lines)


_JS_TEMPLATE = r"""
globalThis.__OUE = globalThis.__OUE || { jobs: {}, device: null, pipe: null };

async function __oueDevice() {
  if (globalThis.__OUE.device) return globalThis.__OUE.device;
  const ad = await navigator.gpu.requestAdapter();
  if (!ad) throw new Error("no WebGPU adapter");
  globalThis.__OUE.device = await ad.requestDevice();
  return globalThis.__OUE.device;
}

const __OUE_WGSL = `
struct U {
  nz: u32, ny: u32, nx: u32, N: u32,
  B: u32, BN: u32, nrx: u32, nsrc: u32,
  step: u32, pad0: u32, pad1: u32, pad2: u32,
  wav: f32, invh: f32, dt: f32, pad3: f32,
};
// field slots in F: 0 vx, 1 vy, 2 vz, 3..8 s1..s6, 9..14 g1..g6
@group(0) @binding(0) var<uniform> P: U;
@group(0) @binding(1) var<storage, read> MAT: array<f32>;   // 21*N C + N rho
@group(0) @binding(2) var<storage, read_write> F: array<f32>;
@group(0) @binding(3) var<storage, read> RECL: array<u32>;
@group(0) @binding(4) var<storage, read_write> REC: array<f32>;
@group(0) @binding(5) var<storage, read> SRC: array<vec4<f32>>; // cell,b,w,-

fn fidx(f: u32, b: u32, cell: u32) -> u32 {
  return f * P.BN + b * P.N + cell;
}

// gamma value at native position, from velocity one-sided differences
fn gamv(j: u32, b: u32, iz: u32, iy: u32, ix: u32) -> f32 {
  let nyx = P.ny * P.nx;
  let cell = iz * nyx + iy * P.nx + ix;
  var v: f32 = 0.0;
  switch j {
    case 0u: {  // gam1 = Db(vx, x)
      if (ix > 0u) {
        v = (F[fidx(0u,b,cell)] - F[fidx(0u,b,cell - 1u)]) * P.invh; }
    }
    case 1u: {  // gam2 = Db(vy, y)
      if (iy > 0u) {
        v = (F[fidx(1u,b,cell)] - F[fidx(1u,b,cell - P.nx)]) * P.invh; }
    }
    case 2u: {  // gam3 = Db(vz, z)
      if (iz > 0u) {
        v = (F[fidx(2u,b,cell)] - F[fidx(2u,b,cell - nyx)]) * P.invh; }
    }
    case 3u: {  // gam4 = Df(vz, y) + Df(vy, z)
      if (iy < P.ny - 1u) {
        v = v + (F[fidx(2u,b,cell + P.nx)] - F[fidx(2u,b,cell)]) * P.invh; }
      if (iz < P.nz - 1u) {
        v = v + (F[fidx(1u,b,cell + nyx)] - F[fidx(1u,b,cell)]) * P.invh; }
    }
    case 4u: {  // gam5 = Df(vz, x) + Df(vx, z)
      if (ix < P.nx - 1u) {
        v = v + (F[fidx(2u,b,cell + 1u)] - F[fidx(2u,b,cell)]) * P.invh; }
      if (iz < P.nz - 1u) {
        v = v + (F[fidx(0u,b,cell + nyx)] - F[fidx(0u,b,cell)]) * P.invh; }
    }
    default: {  // gam6 = Df(vy, x) + Df(vx, y)
      if (ix < P.nx - 1u) {
        v = v + (F[fidx(1u,b,cell + 1u)] - F[fidx(1u,b,cell)]) * P.invh; }
      if (iy < P.ny - 1u) {
        v = v + (F[fidx(0u,b,cell + P.nx)] - F[fidx(0u,b,cell)]) * P.invh; }
    }
  }
  return v;
}

// stored gamma (slot 9+j) moved from its native position by half-cell
// averaging with per-axis validity, matching ringfwi.elastic3d._avg_axis:
// d=+1: g[k] = (f[k]+f[k+s])/2 valid k<n-1;  d=-1: g[k] = (f[k-s]+f[k])/2
// valid k>0;  invalid rows are zero.
fn mvg(j: u32, b: u32, iz: u32, iy: u32, ix: u32,
       dz: i32, dy: i32, dx: i32) -> f32 {
  let nyx = P.ny * P.nx;
  if (dz > 0 && iz >= P.nz - 1u) { return 0.0; }
  if (dz < 0 && iz == 0u) { return 0.0; }
  if (dy > 0 && iy >= P.ny - 1u) { return 0.0; }
  if (dy < 0 && iy == 0u) { return 0.0; }
  if (dx > 0 && ix >= P.nx - 1u) { return 0.0; }
  if (dx < 0 && ix == 0u) { return 0.0; }
  var acc: f32 = 0.0;
  var cnt: f32 = 0.0;
  let z0: i32 = select(0, select(-1, 0, dz > 0), dz != 0);
  let z1: i32 = select(0, select(0, 1, dz > 0), dz != 0);
  let y0: i32 = select(0, select(-1, 0, dy > 0), dy != 0);
  let y1: i32 = select(0, select(0, 1, dy > 0), dy != 0);
  let x0: i32 = select(0, select(-1, 0, dx > 0), dx != 0);
  let x1: i32 = select(0, select(0, 1, dx > 0), dx != 0);
  for (var oz: i32 = z0; oz <= z1; oz = oz + 1) {
    for (var oy: i32 = y0; oy <= y1; oy = oy + 1) {
      for (var ox: i32 = x0; ox <= x1; ox = ox + 1) {
        let cz = u32(i32(iz) + oz);
        let cy = u32(i32(iy) + oy);
        let cx = u32(i32(ix) + ox);
        let cell = cz * nyx + cy * P.nx + cx;
        acc = acc + F[fidx(9u + j, b, cell)];
        cnt = cnt + 1.0;
      }
    }
  }
  return acc / cnt;
}

@compute @workgroup_size(64)
fn strains(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.BN) { return; }
  let b = i / P.N;
  let cell = i % P.N;
  let nyx = P.ny * P.nx;
  let iz = cell / nyx;
  let rem = cell % nyx;
  let iy = rem / P.nx;
  let ix = rem % P.nx;
  for (var j: u32 = 0u; j < 6u; j = j + 1u) {
    F[fidx(9u + j, b, cell)] = gamv(j, b, iz, iy, ix);
  }
}

@compute @workgroup_size(64)
fn stress(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.BN) { return; }
  let b = i / P.N;
  let cell = i % P.N;
  let nyx = P.ny * P.nx;
  let iz = cell / nyx;
  let rem = cell % nyx;
  let iy = rem / P.nx;
  let ix = rem % P.nx;
__STRESS_TERMS__
}

@compute @workgroup_size(64)
fn inject(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= P.nsrc) { return; }
  let e = SRC[k];
  let cell = u32(e.x);
  let b = u32(e.y);
  let a = P.wav * e.z;
  F[fidx(3u, b, cell)] = F[fidx(3u, b, cell)] + a;
  F[fidx(4u, b, cell)] = F[fidx(4u, b, cell)] + a;
  F[fidx(5u, b, cell)] = F[fidx(5u, b, cell)] + a;
}

@compute @workgroup_size(64)
fn velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.BN) { return; }
  let b = i / P.N;
  let cell = i % P.N;
  let nyx = P.ny * P.nx;
  let iz = cell / nyx;
  let rem = cell % nyx;
  let iy = rem / P.nx;
  let ix = rem % P.nx;
  let dtr = P.dt / MAT[21u * P.N + cell];
  var ax: f32 = 0.0;
  var ay: f32 = 0.0;
  var az: f32 = 0.0;
  // vx += dtr * (Df(s1,x) + Db(s6,y) + Db(s5,z))
  if (ix < P.nx - 1u) {
    ax = ax + (F[fidx(3u,b,cell + 1u)] - F[fidx(3u,b,cell)]) * P.invh; }
  if (iy > 0u) {
    ax = ax + (F[fidx(8u,b,cell)] - F[fidx(8u,b,cell - P.nx)]) * P.invh; }
  if (iz > 0u) {
    ax = ax + (F[fidx(7u,b,cell)] - F[fidx(7u,b,cell - nyx)]) * P.invh; }
  // vy += dtr * (Db(s6,x) + Df(s2,y) + Db(s4,z))
  if (ix > 0u) {
    ay = ay + (F[fidx(8u,b,cell)] - F[fidx(8u,b,cell - 1u)]) * P.invh; }
  if (iy < P.ny - 1u) {
    ay = ay + (F[fidx(4u,b,cell + P.nx)] - F[fidx(4u,b,cell)]) * P.invh; }
  if (iz > 0u) {
    ay = ay + (F[fidx(6u,b,cell)] - F[fidx(6u,b,cell - nyx)]) * P.invh; }
  // vz += dtr * (Db(s5,x) + Db(s4,y) + Df(s3,z))
  if (ix > 0u) {
    az = az + (F[fidx(7u,b,cell)] - F[fidx(7u,b,cell - 1u)]) * P.invh; }
  if (iy > 0u) {
    az = az + (F[fidx(6u,b,cell)] - F[fidx(6u,b,cell - P.nx)]) * P.invh; }
  if (iz < P.nz - 1u) {
    az = az + (F[fidx(5u,b,cell + nyx)] - F[fidx(5u,b,cell)]) * P.invh; }
  F[fidx(0u,b,cell)] = F[fidx(0u,b,cell)] + dtr * ax;
  F[fidx(1u,b,cell)] = F[fidx(1u,b,cell)] + dtr * ay;
  F[fidx(2u,b,cell)] = F[fidx(2u,b,cell)] + dtr * az;
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.B * P.nrx) { return; }
  let b = i / P.nrx;
  let j = i % P.nrx;
  let cell = RECL[j];
  let pr = -(F[fidx(3u,b,cell)] + F[fidx(4u,b,cell)] + F[fidx(5u,b,cell)])
           / 3.0;
  REC[(b * P.__NT__u + P.step) * P.nrx + j] = pr;
}
`;

function __oueEnsurePipes(dev, nt) {
  const wgslKey = "nt" + nt;
  globalThis.__OUE.pipes = globalThis.__OUE.pipes || {};
  if (!globalThis.__OUE.pipes[wgslKey]) {
    const code = __OUE_WGSL.replaceAll("P.__NT__u", String(nt) + "u");
    const mod = dev.createShaderModule({ code: code });
    const bgl = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
    ]});
    const lay = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const mkp = (ep) => dev.createComputePipeline({
      layout: lay, compute: { module: mod, entryPoint: ep } });
    globalThis.__OUE.pipes[wgslKey] = {
      bgl: bgl, strains: mkp("strains"), stress: mkp("stress"),
      inject: mkp("inject"), velocity: mkp("velocity"),
      gather: mkp("gather") };
  }
  return globalThis.__OUE.pipes[wgslKey];
}

// Voigt upper-triangle order shared with the Python side (KEYS21).
const __OUE_PAIRS = (() => {
  const p = [];
  for (let i = 0; i < 6; i++) for (let j = i; j < 6; j++) p.push([i, j]);
  return p;
})();

function __oueMul6(A, B) {
  const C = new Float64Array(36);
  for (let i = 0; i < 6; i++)
    for (let k = 0; k < 6; k++) {
      const a = A[i * 6 + k];
      if (a === 0) continue;
      for (let j = 0; j < 6; j++) C[i * 6 + j] += a * B[k * 6 + j];
    }
  return C;
}

// Bond-rotated 6x6 stiffness for a c-axis unit vector
// (R = Rz(azim) Ry(colat); Cp = M C M^T) -- port of the axisfield rotation.
function __oueRot6(base6, ax) {
  const az = Math.max(-1, Math.min(1, ax[2]));
  const t = Math.acos(az), ph = Math.atan2(ax[1], ax[0]);
  const ct = Math.cos(t), st = Math.sin(t);
  const cp = Math.cos(ph), sp = Math.sin(ph);
  const R = [cp * ct, -sp, cp * st,
             sp * ct,  cp, sp * st,
             -st,       0, ct];
  const vp = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]];
  const M = new Float64Array(36);
  for (let a = 0; a < 6; a++) {
    const i = vp[a][0], j = vp[a][1];
    for (let b = 0; b < 6; b++) {
      const k = vp[b][0], l = vp[b][1];
      let v = R[i * 3 + k] * R[j * 3 + l];
      if (k !== l) v += R[i * 3 + l] * R[j * 3 + k];
      M[a * 6 + b] = v;
    }
  }
  const MT = new Float64Array(36);
  for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++)
    MT[a * 6 + b] = M[b * 6 + a];
  return __oueMul6(__oueMul6(M, base6), MT);
}

function __oueRng(seed) {                      // mulberry32
  let s = seed >>> 0;
  return function () {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The whole simulated-annealing loop as ONE main-thread job: proposals,
// soft-Voronoi model building (JS), batched elastic forward (GPU), RX-filter
// + weighted residual (JS), Metropolis -- no Streamlit rerun tax per step.
globalThis.__oueSA = function (id, cfg) {
  const job = { done: false, prog: 0, total: cfg.steps, error: null,
                result: null };
  globalThis.__OUE.jobs[id] = job;
  const F32 = (b) => new Float32Array(
    ((b instanceof Uint8Array) ? b.slice().buffer
                               : new Uint8Array(b).slice().buffer));
  const U32 = (b) => new Uint32Array(
    ((b instanceof Uint8Array) ? b.slice().buffer
                               : new Uint8Array(b).slice().buffer));
  (async () => {
    try {
      const dev = await __oueDevice();
      const nz = cfg.nz, ny = cfg.ny, nx = cfg.nx, nt = cfg.nt;
      const B = cfg.B, nrx = cfg.nrx, G = cfg.G;
      const N = nz * ny * nx, BN = B * N;
      const matBase = F32(cfg.matBase);
      const cells = U32(cfg.cells);
      const px = F32(cfg.px), py = F32(cfg.py), pz = F32(cfg.pz);
      const base6 = new Float64Array(F32(cfg.base6));
      const wav = F32(cfg.wav);
      const recLin = U32(cfg.recLin);
      const src = F32(cfg.src);
      const dobs = F32(cfg.dobs);
      const W = F32(cfg.W);
      const sos = F32(cfg.sos);
      const P = cells.length;
      const s2 = 2.0 * (cfg.tau * cfg.h) * (cfg.tau * cfg.h);

      const pipe = __oueEnsurePipes(dev, nt);
      const mk = (sz, usage) => dev.createBuffer({ size: sz, usage: usage });
      const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const bufMat = mk(matBase.byteLength, ST);
      const bufF = mk(15 * BN * 4, ST | GPUBufferUsage.COPY_SRC);
      const bufRecL = mk(Math.max(nrx, 1) * 4, ST);
      dev.queue.writeBuffer(bufRecL, 0, recLin);
      const bufRec = mk(B * nt * nrx * 4, ST | GPUBufferUsage.COPY_SRC);
      const bufSrc = mk(Math.max(src.length / 4, 1) * 16, ST);
      dev.queue.writeBuffer(bufSrc, 0, src);
      const uStride = 256;
      const bufU = dev.createBuffer({ size: nt * uStride,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const staging = dev.createBuffer({ size: B * nt * nrx * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const bg = dev.createBindGroup({ layout: pipe.bgl, entries: [
        { binding: 0, resource: { buffer: bufU, size: 64 } },
        { binding: 1, resource: { buffer: bufMat } },
        { binding: 2, resource: { buffer: bufF } },
        { binding: 3, resource: { buffer: bufRecL } },
        { binding: 4, resource: { buffer: bufRec } },
        { binding: 5, resource: { buffer: bufSrc } },
      ]});
      const uData = new ArrayBuffer(nt * uStride);
      for (let n = 0; n < nt; n++) {
        const dv = new DataView(uData, n * uStride, 64);
        dv.setUint32(0, nz, true); dv.setUint32(4, ny, true);
        dv.setUint32(8, nx, true); dv.setUint32(12, N, true);
        dv.setUint32(16, B, true); dv.setUint32(20, BN, true);
        dv.setUint32(24, nrx, true);
        dv.setUint32(28, src.length / 4, true);
        dv.setUint32(32, n, true);
        dv.setFloat32(48, wav[n], true);
        dv.setFloat32(52, cfg.invh, true);
        dv.setFloat32(56, cfg.dt, true);
      }
      dev.queue.writeBuffer(bufU, 0, uData);

      const mat = matBase.slice();
      const Cg = [];
      const w = new Float64Array(G);

      function buildMat(seeds, axes) {
        for (let g = 0; g < G; g++) Cg[g] = __oueRot6(base6, axes[g]);
        for (let q = 0; q < P; q++) {
          const cx = px[q], cy = py[q], cz = pz[q];
          let mx = -Infinity;
          for (let g = 0; g < G; g++) {
            const dx = cx - seeds[g][0], dy = cy - seeds[g][1],
                  dz = cz - seeds[g][2];
            const a = -(dx * dx + dy * dy + dz * dz) / s2;
            w[g] = a;
            if (a > mx) mx = a;
          }
          let tot = 0;
          for (let g = 0; g < G; g++) {
            w[g] = Math.exp(w[g] - mx); tot += w[g];
          }
          const cell = cells[q];
          for (let k = 0; k < 21; k++) {
            const i = __OUE_PAIRS[k][0], j = __OUE_PAIRS[k][1];
            let v = 0;
            for (let g = 0; g < G; g++) v += w[g] * Cg[g][i * 6 + j];
            mat[k * N + cell] = v / tot;
          }
        }
      }

      const wgF = Math.ceil(BN / 64);
      const wgS = Math.ceil(Math.max(src.length / 4, 1) / 64);
      const wgR = Math.ceil(Math.max(B * nrx, 1) / 64);

      async function forward() {
        dev.queue.writeBuffer(bufMat, 0, mat);
        const enc = dev.createCommandEncoder();
        enc.clearBuffer(bufF);
        enc.clearBuffer(bufRec);
        const pass = enc.beginComputePass();
        for (let n = 0; n < nt; n++) {
          const off = [n * uStride];
          pass.setPipeline(pipe.strains); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgF);
          pass.setPipeline(pipe.stress); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgF);
          pass.setPipeline(pipe.inject); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgS);
          pass.setPipeline(pipe.velocity); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgF);
          pass.setPipeline(pipe.gather); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgR);
        }
        pass.end();
        enc.copyBufferToBuffer(bufRec, 0, staging, 0, B * nt * nrx * 4);
        dev.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return out;
      }

      const nSec = sos.length / 6;
      function J_of(tr) {
        let J = 0;
        for (let b = 0; b < B; b++) {
          for (let r = 0; r < nrx; r++) {
            const wt = W[b * nrx + r];
            if (wt === 0) continue;
            const z = [];
            for (let sct = 0; sct < nSec; sct++) z.push([0, 0]);
            let acc = 0;
            for (let n = 0; n < nt; n++) {
              let x = tr[(b * nt + n) * nrx + r];
              for (let sct = 0; sct < nSec; sct++) {
                const o = sct * 6;
                const y = sos[o] * x + z[sct][0];
                z[sct][0] = sos[o + 1] * x - sos[o + 4] * y + z[sct][1];
                z[sct][1] = sos[o + 2] * x - sos[o + 5] * y;
                x = y;
              }
              const d = (x - dobs[(b * nt + n) * nrx + r]) * wt;
              acc += d * d;
            }
            J += acc;
          }
        }
        return 0.5 * J;
      }

      const rng = __oueRng(cfg.rngSeed);
      const seeds = [], axes = [];
      const s0 = F32(cfg.seeds0), a0 = F32(cfg.axes0);
      for (let g = 0; g < G; g++) {
        seeds.push([s0[3 * g], s0[3 * g + 1], s0[3 * g + 2]]);
        axes.push([a0[3 * g], a0[3 * g + 1], a0[3 * g + 2]]);
      }
      const clone = (arr) => arr.map(v => v.slice());
      const norm3 = (v) => {
        const n_ = Math.hypot(v[0], v[1], v[2]);
        v[0] /= n_; v[1] /= n_; v[2] /= n_;
        if (v[2] < 0) { v[0] = -v[0]; v[1] = -v[1]; v[2] = -v[2]; }
        return v;
      };
      const gauss = () => {
        const u = Math.max(rng(), 1e-12), v2 = rng();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v2);
      };

      buildMat(seeds, axes);
      const J0 = J_of(await forward());
      let Jc = J0, bJ = J0;
      let bSeeds = clone(seeds), bAxes = clone(axes);
      const hist = [J0];

      for (let k = 0; k < cfg.steps; k++) {
        const sT = clone(seeds), aT = clone(axes);
        const g = Math.floor(rng() * G);
        const u = rng();
        if (u < 0.30) {
          const a = aT[g];
          a[0] += 0.26 * gauss(); a[1] += 0.26 * gauss();
          a[2] += 0.26 * gauss();
          norm3(a);
        } else if (u < 0.50) {
          aT[g] = norm3([gauss(), gauss(), Math.abs(gauss())]);
        } else if (u < 0.75) {
          sT[g][0] += 1.5 * cfg.h * gauss();
          sT[g][1] += 1.5 * cfg.h * gauss();
          sT[g][2] += 1.5 * cfg.h * gauss();
        } else if (u < 0.85) {
          const q = Math.floor(rng() * P);
          sT[g] = [px[q], py[q], pz[q]];
        } else {
          const g2 = Math.floor(rng() * G);
          const tmp = aT[g]; aT[g] = aT[g2]; aT[g2] = tmp;
        }
        buildMat(sT, aT);
        const Jt = J_of(await forward());
        const T = J0 * 0.3 * Math.pow(1e-3 / 0.3,
                                      k / Math.max(cfg.steps - 1, 1));
        if (Jt < Jc || rng() < Math.exp(-(Jt - Jc) / Math.max(T, 1e-30))) {
          for (let g2 = 0; g2 < G; g2++) {
            seeds[g2] = sT[g2]; axes[g2] = aT[g2];
          }
          Jc = Jt;
        }
        if (Jt < bJ) {
          bJ = Jt; bSeeds = clone(sT); bAxes = clone(aT);
          hist.push(Jt);
        }
        job.prog = k + 1;
      }

      const out = new Float32Array(3 + 6 * G + hist.length);
      out[0] = bJ; out[1] = J0; out[2] = hist.length;
      for (let g = 0; g < G; g++) {
        out[3 + 3 * g] = bSeeds[g][0];
        out[3 + 3 * g + 1] = bSeeds[g][1];
        out[3 + 3 * g + 2] = bSeeds[g][2];
        out[3 + 3 * G + 3 * g] = bAxes[g][0];
        out[3 + 3 * G + 3 * g + 1] = bAxes[g][1];
        out[3 + 3 * G + 3 * g + 2] = bAxes[g][2];
      }
      out.set(Float32Array.from(hist), 3 + 6 * G);
      for (const bf of [bufMat, bufF, bufRecL, bufRec, bufSrc, bufU,
                        staging])
        bf.destroy();
      job.result = out;
      job.done = true;
    } catch (e) {
      job.error = String(e);
      job.done = true;
    }
  })();
};

globalThis.__oueStart = function (id, nz, ny, nx, B, nt, invh, dt,
                                  matB, wavB, recLinB, srcB, nrx, nsrc) {
  const job = { done: false, prog: 0, total: nt, error: null, result: null };
  globalThis.__OUE.jobs[id] = job;
  const asU8 = (b) => (b instanceof Uint8Array) ? b : new Uint8Array(b);
  (async () => {
    try {
      const dev = await __oueDevice();
      const N = nz * ny * nx;
      const BN = B * N;
      const mat = new Float32Array(asU8(matB).slice().buffer);
      const wav = new Float32Array(asU8(wavB).slice().buffer);
      const recLin = new Uint32Array(asU8(recLinB).slice().buffer);
      const src = new Float32Array(asU8(srcB).slice().buffer);

      const mk = (sz, usage) => dev.createBuffer({ size: sz, usage: usage });
      const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const bufMat = mk(mat.byteLength, ST);
      dev.queue.writeBuffer(bufMat, 0, mat);
      const bufF = mk(15 * BN * 4, ST | GPUBufferUsage.COPY_SRC);
      const bufRecL = mk(Math.max(nrx, 1) * 4, ST);
      dev.queue.writeBuffer(bufRecL, 0, recLin);
      const bufRec = mk(B * nt * nrx * 4, ST | GPUBufferUsage.COPY_SRC);
      const bufSrc = mk(Math.max(nsrc, 1) * 16, ST);
      dev.queue.writeBuffer(bufSrc, 0, src);
      const uStride = 256;
      const bufU = dev.createBuffer({
        size: nt * uStride,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const staging = dev.createBuffer({
        size: B * nt * nrx * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      const pipe = __oueEnsurePipes(dev, nt);
      const bg = dev.createBindGroup({ layout: pipe.bgl, entries: [
        { binding: 0, resource: { buffer: bufU, size: 64 } },
        { binding: 1, resource: { buffer: bufMat } },
        { binding: 2, resource: { buffer: bufF } },
        { binding: 3, resource: { buffer: bufRecL } },
        { binding: 4, resource: { buffer: bufRec } },
        { binding: 5, resource: { buffer: bufSrc } },
      ]});

      const uData = new ArrayBuffer(nt * uStride);
      for (let n = 0; n < nt; n++) {
        const dv = new DataView(uData, n * uStride, 64);
        dv.setUint32(0, nz, true); dv.setUint32(4, ny, true);
        dv.setUint32(8, nx, true); dv.setUint32(12, N, true);
        dv.setUint32(16, B, true); dv.setUint32(20, BN, true);
        dv.setUint32(24, nrx, true); dv.setUint32(28, nsrc, true);
        dv.setUint32(32, n, true);
        dv.setFloat32(48, wav[n], true);
        dv.setFloat32(52, invh, true);
        dv.setFloat32(56, dt, true);
      }
      dev.queue.writeBuffer(bufU, 0, uData);

      const wgF = Math.ceil(BN / 64);
      const wgS = Math.ceil(Math.max(nsrc, 1) / 64);
      const wgR = Math.ceil(Math.max(B * nrx, 1) / 64);
      // Chunked submission: real progress between slices (a single opaque
      // submission shows nothing while the GPU works).
      const CH = Math.max(1, Math.ceil(nt / 12));
      for (let n0 = 0; n0 < nt; n0 += CH) {
        const enc = dev.createCommandEncoder();
        if (n0 === 0) { enc.clearBuffer(bufF); enc.clearBuffer(bufRec); }
        const pass = enc.beginComputePass();
        const nEnd = Math.min(n0 + CH, nt);
        for (let n = n0; n < nEnd; n++) {
          const off = [n * uStride];
          pass.setPipeline(pipe.strains); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgF);
          pass.setPipeline(pipe.stress); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgF);
          pass.setPipeline(pipe.inject); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgS);
          pass.setPipeline(pipe.velocity); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgF);
          pass.setPipeline(pipe.gather); pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroups(wgR);
        }
        pass.end();
        if (nEnd >= nt) {
          enc.copyBufferToBuffer(bufRec, 0, staging, 0, B * nt * nrx * 4);
        }
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
        job.prog = nEnd;
        try {
          globalThis.__ouProgChan = globalThis.__ouProgChan
            || new BroadcastChannel('ou_progress');
          globalThis.__ouProgChan.postMessage({
            frac: nEnd / nt,
            text: 'Elastic solve on YOUR GPU (WebGPU): step '
                  + nEnd + '/' + nt });
        } catch (e) {}
      }
      await staging.mapAsync(GPUMapMode.READ);
      job.result = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      for (const bf of [bufMat, bufF, bufRecL, bufRec, bufSrc, bufU, staging])
        bf.destroy();
      job.prog = nt;
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

# Worker-side bridge: stlite parks the Python worker so async WebGPU never
# advances THERE, but message events demonstrably reach it (the kernel's own
# traffic). The GPU engines therefore live on the MAIN thread (injected into
# index.html at build time); this side only posts job requests and receives
# progress/results over a BroadcastChannel.
_BRIDGE_JS = r"""
globalThis.__OUB = globalThis.__OUB || (() => {
  const st = { jobs: {}, ch: new BroadcastChannel('ou_gpu') };
  const handle = (m) => {
    if (!m || !m.id) return;
    const j = st.jobs[m.id];
    if (!j) return;
    if (m.type === 'prog') { j.prog = m.prog; j.total = m.total; }
    if (m.type === 'done') {
      j.done = true;
      j.error = m.error || null;
      if (m.result) { j.result = new Float32Array(m.result); }
      j.prog = j.total;
    }
  };
  // Results arrive as DIRECT worker messages (the only events this parked
  // worker reliably dispatches); the BC listener is kept as a fallback.
  self.addEventListener('message', (ev) => {
    const m = ev.data;
    if (m && m.__ougpu) { handle(m); }
  });
  st.ch.onmessage = (ev) => handle(ev.data);
  return st;
})();
globalThis.__oubStart = function (id, msg) {
  globalThis.__OUB.jobs[id] = { done: false, prog: 0, total: 0,
                                error: null, result: null };
  globalThis.__OUB.ch.postMessage(msg);
};
"""


def available():
    """True inside a browser (Pyodide) whose runtime exposes WebGPU."""
    try:
        import js
        return getattr(js.navigator, "gpu", None) is not None
    except Exception:
        return False


def _js_source():
    """Main-thread GPU engine source (embedded in index.html at build)."""
    return _JS_TEMPLATE.replace("__STRESS_TERMS__", _gen_stress_terms())


def _ensure_js():
    global _loaded
    if not _loaded:
        import js
        js.eval(_BRIDGE_JS)
        _loaded = True


def start(Cmaps, rho, h, dt, nt, wavelet, src_pts_list, rec_idx):
    """Launch a batched elastic FMC on the client GPU; returns a job id."""
    import js
    from pyodide.ffi import to_js

    _ensure_js()
    rho = np.asarray(rho, float)
    nz, ny, nx = rho.shape
    N = nz * ny * nx
    B = len(src_pts_list)

    mat = np.empty((22, N), np.float32)
    for k, key in enumerate(KEYS21):
        v = Cmaps[key]
        mat[k] = (np.full(N, float(v), np.float32) if np.ndim(v) == 0
                  else np.asarray(v, np.float32).ravel())
    mat[21] = rho.astype(np.float32).ravel()

    def lin(t):
        return (t[0] * ny + t[1]) * nx + t[2]

    rec_lin = np.array([lin(t) for t in rec_idx], dtype=np.uint32)
    src = []
    for b, pts in enumerate(src_pts_list):
        for idx, w in pts:
            src.append([float(lin(idx)), float(b), float(w), 0.0])
    src = np.asarray(src, np.float32)

    _counter[0] += 1
    job_id = f"el{_counter[0]}"
    msg = to_js(dict(
        type="el_start", id=job_id, nz=nz, ny=ny, nx=nx, B=int(B),
        nt=int(nt), invh=float(1.0 / h), dt=float(dt),
        mat=mat.tobytes(),
        wav=np.asarray(wavelet, np.float32).tobytes(),
        recLin=rec_lin.tobytes(), src=src.tobytes(),
        nrx=int(len(rec_lin)), nsrc=int(len(src))),
        dict_converter=js.Object.fromEntries)
    js.__oubStart(job_id, msg)
    return job_id


def poll(job_id):
    """{'done': bool, 'prog': int, 'total': int, 'error': str|None}."""
    import js
    job = getattr(js.__OUB.jobs, job_id)
    return dict(done=bool(job.done), prog=int(job.prog),
                total=int(job.total),
                error=(str(job.error) if job.error else None))


def result(job_id, B, nt, n_rx):
    """Fetch the finished batched FMC as (B, nt, n_rx) float64."""
    import js
    job = getattr(js.__OUB.jobs, job_id)
    buf = np.asarray(job.result.to_py(), dtype=np.float32)
    js.eval("delete globalThis.__OUB.jobs['" + job_id + "']")
    return buf.reshape(B, nt, n_rx).astype(np.float64)


def sa_start(shape, mat_base, cells, px, py, pz, base6, h, dt, nt, tau,
             wavelet, src_cells, rec_lin, dobs, W, sos, seeds0, axes0,
             steps, rng_seed):
    """Launch the ENTIRE annealing loop as one main-thread job.

    The main thread does proposals, soft-Voronoi model building, batched GPU
    forwards, RX-filtered weighted residuals and Metropolis acceptance -- no
    Streamlit rerun per step. Returns a job id; poll as usual and fetch the
    packed result with :func:`sa_result`.
    """
    import js
    from pyodide.ffi import to_js

    _ensure_js()
    nz, ny, nx = (int(v) for v in shape)
    src = np.asarray([[float(c), float(b), 1.0, 0.0]
                      for b, c in enumerate(src_cells)], np.float32)
    G = int(np.asarray(seeds0).size // 3)
    _counter[0] += 1
    job_id = f"sa{_counter[0]}"
    msg = to_js(dict(
        type="sa_start", id=job_id, nz=nz, ny=ny, nx=nx,
        B=int(len(src_cells)), nt=int(nt), nrx=int(len(rec_lin)), G=G,
        invh=float(1.0 / h), dt=float(dt), h=float(h), tau=float(tau),
        steps=int(steps), rngSeed=int(rng_seed) & 0x7FFFFFFF,
        matBase=np.asarray(mat_base, np.float32).ravel().tobytes(),
        cells=np.asarray(cells, np.uint32).tobytes(),
        px=np.asarray(px, np.float32).tobytes(),
        py=np.asarray(py, np.float32).tobytes(),
        pz=np.asarray(pz, np.float32).tobytes(),
        base6=np.asarray(base6, np.float32).ravel().tobytes(),
        wav=np.asarray(wavelet, np.float32).tobytes(),
        recLin=np.asarray(rec_lin, np.uint32).tobytes(),
        src=src.tobytes(),
        dobs=np.asarray(dobs, np.float32).ravel().tobytes(),
        W=np.asarray(W, np.float32).ravel().tobytes(),
        sos=np.asarray(sos, np.float32).ravel().tobytes()),
        dict_converter=js.Object.fromEntries)
    js.__oubStart(job_id, msg)
    return job_id


def sa_result(job_id):
    """Fetch a finished annealing job's packed result (flat float32)."""
    import js
    job = getattr(js.__OUB.jobs, job_id)
    buf = np.asarray(job.result.to_py(), dtype=np.float32)
    js.eval("delete globalThis.__OUB.jobs['" + job_id + "']")
    return buf
