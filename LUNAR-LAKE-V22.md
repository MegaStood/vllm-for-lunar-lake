# Running vLLM 0.22 on Intel Lunar Lake (Arc 140V iGPU)

This `v22-lunar-lake` branch carries the patches and the recipe to run
**vLLM 0.22.0** on an Intel **Lunar Lake** handheld. Developed and verified
on an **MSI Claw 8 AI+** — Core Ultra 7 258V, Arc 140V iGPU (Xe2-LPG),
32 GB shared LPDDR5X.

This branch is the v0.22 sibling of [`lunar-lake`](../../tree/lunar-lake)
(which is the v0.21 recipe). Use this one if you want INT4 MoE with the
new **oracle backend** (`XPUExpertsInt4` via CUTLASS-SYCL), which is what
unblocks **Nemotron-3-Nano-30B-A3B-AWQ** and **LFM2-24B-A2B** on iGPU —
the v0.21 path could not handle their non-256-aligned tensor shapes.

---

## Status (2026-06-13) — serves end to end

| Model | Quant | MoE path | Notes |
|---|---|---|---|
| **Nemotron-3-Nano-30B-A3B-AWQ** | compressed-tensors WNA16 sym-INT4 | XPU (CUTLASS-SYCL) | hybrid Mamba2+attn, primary OpenClaw daily driver. Warm prefill 1673 tok/s, decode 22 tok/s at 7K. |
| **LFM2-24B-A2B** | compressed-tensors WNA16 sym-INT4 | XPU (CUTLASS-SYCL) | flat decode across full 32K (22→17 tok/s), no decode-wall |
| **LFM2.5-8B FP8** | per-channel FP8 (compressed-tensors) | Triton fallback | reasoning model — emits `<think>` blocks |
| **Gemma 4 26B QAT** | W4A16 compressed-tensors | **TRITON fallback** (XPUExpertsInt4 rejects GELU_TANH) | drafter-spec model also available; PR #38891 vendored in this branch unblocks per-layer FlashAttention (~83% of layers — see patch 5) |
| **GLM-4.7-Flash** | compressed-tensors | XPU | needs MLA `kv_b_proj` quantized-dtype fix (same as v0.21 fork's GLM patch — applied to the equivalent file) |
| **Qwen3.5-35B-A3B (GPTQ-int4)** | GPTQ-INT4 | XPU | 4-line `_process_weights_xpu` tuple-shape fix needed; perf bottlenecked by GDN Triton fallback |
| **gpt-oss-20b** | MXFP4 | XPU | (carry-over from v0.21 recipe) |
| **qwen3-coder-30B** | compressed-tensors WNA16 sym-INT4 | XPU | works but production daily-driver stays on v0.19 — see project history |

### Not yet working / structurally blocked

- **Nemotron-3-Nano-30B-A3B Mamba2** — RESOLVED with this branch; was blocker on v0.21 due to shape-overfit `_apply_xpu_int4` in v19-backport
- **qwen3.5-35B Autoround** — INC quant doesn't claim MoE → falls to bf16 → kernel OOM
- **benchmark_moe.py autotuner** — structurally CUDA-only (CUDAGraph + Ray "GPU" key); the LFM2-registration patch in this branch makes the script *get past* the architecture-recognition gate but doesn't make it actually tune on XPU

---

## TL;DR — the same trap, plus one new install-time gotcha

`sycl::errc::backend_mismatch` and opaque `libsycl.so.*` errors on Intel XPU
are **almost always an oneAPI _version_ problem — not a vLLM code bug.**
Before you touch a line of vLLM source:

1. `ls -la /opt/intel/oneapi/compiler/latest` — must point at `2025.3`.
2. Is the launcher sourcing `setvars.sh`? **It shouldn't** — see *Environment*.
3. `readelf -d` the failing `.so` — which `libsycl` SONAME does it want?
4. Stale JIT caches: `~/.triton/cache`, the SYCL persistent cache.

**New gotcha for v0.22:** the install pulls
**`setuptools-rust>=1.9.0`** (not needed for v0.21). And with pip 26+, the
PyTorch XPU index is strict — you'll need `--extra-index-url
https://pypi.org/simple` so torch's runtime dependencies
(`intel-cmplr-lib-rt`, `intel-sycl-rt`, `intel-opencl-rt`, etc.) resolve from
PyPI.

---

## The stack

| Component | Version | Notes |
|---|---|---|
| vLLM | 0.22.0 + PR #42032 + the 4 patches in this branch | |
| torch | 2.11.0+xpu | from `https://download.pytorch.org/whl/xpu` |
| vllm-xpu-kernels | **0.1.8** — stock GitHub release | no AOT rebuild needed |
| triton-xpu | 3.7.0 | source-install from the v0.21 fork venv |
| oneAPI | 2025.3 | **build-time only** |
| OS | Linux (developed on Nobara / Fedora 43) | |

---

## The patches (this branch vs. vanilla v0.22.0)

The branch is based on vanilla v0.22.0 with:

### 0. PR #42032 vendored as commit `0fd604c` (upstream, by jinyouzhi)

`[Feat][XPU] Implement WNA16 MoE backend support with INT4 quantization`.
Adds the `XPUExpertsInt4` class (CUTLASS-SYCL kernel) — the real INT4 MoE
XPU wiring that the v0.21 fork was missing. Still draft upstream as of
2026-06-13; see
[vllm-project/vllm#42032](https://github.com/vllm-project/vllm/pull/42032).
A standalone `.patch` file for this commit is also vendored at
`patches/0001-pr-42032-xpu-int4-moe.patch` for users who want to apply it
themselves to a fresh v0.22.0 checkout.

### 1. `vllm/platforms/xpu.py` — `is_integrated_gpu()` via PCI ID set

Same as the v0.21 fork. Without this, vLLM aborts at startup because
`torch.xpu.mem_get_info()` underreports free memory on UMA iGPUs.
Backport of upstream **PR #40295** (still open).

### 2. TritonWNA16Experts wire-up — XPU fallback for GELU_TANH

`XPUExpertsInt4` only supports SwiGLU-family activations. Gemma 4 uses
`GELU_TANH` and would otherwise fail to load on XPU. This patch:

- Adds `WNA16MoEBackend.TRITON` and updates XPU priority to `[XPU, TRITON]`
  in `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
- Replaces the `NotImplementedError` stubs in `TritonWNA16Experts` with
  real implementations in
  `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`

LFM2-24B and Nemotron-3-Nano keep using the XPU CUTLASS path; only Gemma 4
QAT falls through to the Triton path.

### 3. `compressed_tensors_moe_wna16.py` — call `_expert_routing_tables()`

Upstream renamed `_maybe_init_expert_routing_tables` to
`_expert_routing_tables` on `FusedMoE` but missed updating the
compressed-tensors callsite. Without this fix, XPU INT4 MoE loads
AttributeError at kernel-prepare time.

### 4. `benchmarks/kernels/benchmark_moe.py` — register `Lfm2MoeForCausalLM`

The benchmark's architecture-recognition switch didn't include LFM2 MoE.
This adds it so `benchmark_moe.py` gets past the model-config check.

(NOTE: the autotuner itself is still structurally CUDA-only on XPU — see
`project_vllm_benchmark_moe_xpu_unusable`. This patch only fixes the
known-models gap.)

### 5. `vllm/model_executor/models/config.py` — Gemma 4 per-layer attention backend

Vendors upstream **PR #38891** (still OPEN, `needs-rebase` as of 2026-06-13).
Fixes issue [#38887](https://github.com/vllm-project/vllm/issues/38887): on
v0.22.0 stock, `Gemma4Config.verify_and_update_config` forces TRITON_ATTN
globally whenever the model has heterogeneous head dimensions (head_dim=256
sliding + global_head_dim=512 full). This penalises the ~83% of Gemma 4
layers that *can* use the fast FlashAttention path (head_dim=256 ≤ FA's
kernel limit).

This patch removes the global force and lets vLLM's per-layer backend
selector route each layer to the best backend it supports:

| Layer type | head_dim | Backend (XPU) |
|---|---|---|
| Sliding-window (~83%) | 256 | XPU FlashAttention (fast) |
| Full-attention (~17%) | 512 | TRITON_ATTN (fallback) |

Expected impact on Lunar Lake / Arc 140V: Gemma 4 26B QAT decode at 16K
KV should move from ~6 tok/s up to ~12-15 tok/s. Prefill should improve
similarly. Still below LFM2-24B (17 tok/s flat across 32K) because the
17% full-attention layers are still Triton-bound.

NOTE: This patch is verbatim port of PR #38891's diff. The upstream
status is "needs-rebase" so users applying the standalone patch file
should re-derive against their target vLLM tree.

---

## Install recipe

See [`~/install-v22-stock.sh`](../install-v22-stock.sh) (referenced via
the developer's home dir; reproduce in your own clone). The skeleton:

```bash
# Build deps (NEW vs v0.21: setuptools-rust)
pip install --upgrade pip
pip install --upgrade 'setuptools<81' wheel setuptools_scm \
  'setuptools-rust>=1.9.0' 'cmake>=3.26.1' ninja \
  --extra-index-url https://pypi.org/simple

# Torch 2.11.0+xpu
pip install torch==2.11.0+xpu torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://pypi.org/simple

# vllm-xpu-kernels 0.1.8 (stock wheel)
pip install \
  https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.8/vllm_xpu_kernels-0.1.8-cp38-abi3-manylinux_2_28_x86_64.whl

# triton-xpu source-install from the v0.21 fork venv
# (no public wheel; reuse triton-xpu 3.7.0 from ~/vllm-ll-venv/lib/.../triton)

# Editable vllm install from this branch
git clone -b v22-lunar-lake https://github.com/MegaStood/vllm-for-lunar-lake.git ~/vllm-v22
cd ~/vllm-v22
source /opt/intel/oneapi/setvars.sh --force   # BUILD ONLY
VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation
```

Then **scrub the oneAPI env from your runtime shell** — the launchers in
`launchers/` do this for you.

---

## Environment — get this right or nothing works

### Do NOT `source setvars.sh` to *run* vLLM

`setvars.sh` injects `/opt/intel/oneapi/compiler/2025.3/lib` (or whatever
`latest` points at) into `LD_LIBRARY_PATH`. Torch's bundled `libsycl.so.9`
is 2025.3.2, but the system 2025.3 build is slightly different — they
collide and produce `backend_mismatch` errors that look like vLLM bugs.

The launchers in `launchers/` explicitly **unset** all `ONEAPI_*` and
`CMPLR_*` env vars and remove `/opt/intel/oneapi/*` from
`LD_LIBRARY_PATH`. Adapt them to your paths but **keep the scrubbing**.

### oneAPI `latest` symlink must point at 2025.3 (NOT 2026.0)

```bash
ls -la /opt/intel/oneapi/compiler/latest
# → 2025.3
```

If you've installed oneAPI 2026.0, the symlink probably auto-pointed at
it. Repoint it back to 2025.3 or the AOT step at build time will mismatch
torch's bundled SYCL runtime.

### Clear `~/.triton/cache` after any toolchain change

Triton JIT-caches compiled artifacts against a specific `libsycl.so.N`
ABI. After any oneAPI symlink change or torch upgrade:

```bash
rm -rf ~/.triton/cache ~/.cache/sycl-vllm-xpu-v22 ~/.cache/neo-vllm-xpu-v22
```

---

## Launchers

See [`launchers/`](./launchers/) for the per-model serve scripts used in
development. They reference user-specific paths (`/shared/models/...`,
`~/vllm-v22-venv`); adapt to your layout. The pattern is consistent
across all of them:

1. Scrub oneAPI env vars and `LD_LIBRARY_PATH`
2. Activate the v0.22 venv
3. Set per-launcher SYCL/NEO cache dirs (avoid clashing with the v0.21 fork's caches)
4. `exec vllm serve <model> --device xpu ...`

The Nemotron launcher (`vllm-nemotron-3-nano-v22`) is the recommended
starting point — it's the most-stress-tested config on this hardware.

---

## Cross-references

- v0.21 sibling: [`lunar-lake`](../../tree/lunar-lake) branch + `LUNAR-LAKE.md`
- PR #42032 upstream: https://github.com/vllm-project/vllm/pull/42032
- PR #40295 upstream (still open): https://github.com/vllm-project/vllm/pull/40295
- Related upstream blocker: https://github.com/vllm-project/vllm/issues/38887 (TRITON_ATTN head-dim)
