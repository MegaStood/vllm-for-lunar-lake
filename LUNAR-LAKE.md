# Running vLLM on Intel Lunar Lake (Arc 140V iGPU)

This `lunar-lake` branch carries the patches and the recipe to run **vLLM 0.21.0**
on an Intel **Lunar Lake** handheld. Developed and verified on an **MSI Claw 8
AI+** — Core Ultra 7 258V, Arc 140V iGPU (Xe2-LPG), 32 GB shared LPDDR5X.

**Status (2026-05-22) — serves, end to end:**

| Model | MoE path | Smoke test |
|---|---|---|
| gpt-oss-20b | MXFP4 | `17 × 23` → `391` |
| qwen3-coder-30B | INT4 compressed-tensors | reverse-a-string → `s[::-1]` |

Both run on **stock `vllm-xpu-kernels`** — no custom kernel build required.

---

## TL;DR — the one trap that cost a week

`sycl::errc::backend_mismatch` and opaque `libsycl.so.*` errors on Intel XPU are
**almost always an oneAPI _version_ problem — not a vLLM code bug.** Before you
touch a line of vLLM source:

1. `ls -la /opt/intel/oneapi/compiler/latest` — what does it point at?
2. Is the launcher sourcing `setvars.sh`? (It shouldn't — see *Environment*.)
3. `readelf -d` the failing `.so` — which `libsycl` SONAME does it want?
4. Stale JIT caches: `~/.triton/cache`, the SYCL persistent cache.

A week went into AOT kernel rebuilds, an oneDNN patch and `libsycl` swaps,
chasing a "code" bug that never existed. The real cause was **two oneAPI
versions on the box plus a stale Triton cache**. Audit the toolchain first.

---

## The stack

| Component | Version | Notes |
|---|---|---|
| vLLM | 0.21.0 + the 3 patches in this branch | |
| torch | 2.11.0+xpu | from `https://download.pytorch.org/whl/xpu` |
| vllm-xpu-kernels | **0.1.8.1** — stock GitHub release | no AOT rebuild needed |
| triton-xpu | 3.7.0 | not on PyPI — see *Install* |
| oneAPI | 2025.3 | **build-time only** |
| OS | Linux (developed on Nobara / Fedora 43) | |

---

## The patches (this branch vs. vanilla v0.21.0)

Three additive patches, ~140 lines total.

### 1. `vllm/platforms/xpu.py` — implement `is_integrated_gpu()`
`MemorySnapshot` uses host-RAM accounting for integrated (UMA) GPUs, gated on
`Platform.is_integrated_gpu()`. `XPUPlatform` never implemented it → returned the
base default `False` → on a UMA iGPU, `torch.xpu.mem_get_info()` underreports
free memory (it ignores reclaimable page cache), so vLLM aborts at startup with
*"Free memory on device xpu:0 is less than desired GPU memory utilization."*

The patch implements `is_integrated_gpu()` by matching the PCI device ID
(`torch.xpu.get_device_properties().device_id`) against a curated set of Intel
iGPU IDs — Lunar Lake `0x64A0`, Arrow Lake-H `0x7D51/0x7DD1`, Panther Lake
`0xB0xx`. (`torch.xpu` exposes no `is_integrated` flag, unlike CUDA.)
Upstreamed as **vLLM PR #40295**.

### 2. `vllm/v1/worker/xpu_worker.py` — skip the `world_size==1` warmup all-reduce
The XCCL warmup all-reduce triggers a cold SYCL-JIT crash on Lunar Lake, and at
TP=1 it is a no-op. The patch skips it when `world_size == 1`.

### 3. `compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py` — XPU INT4 MoE
Upstream's INT4 compressed-tensors MoE path calls CUDA-only
`gptq_marlin_moe_repack`. The patch adds an XPU branch: convert the int32-packed
GPTQ weights to uint8 two's-complement and route the forward through
`vllm_xpu_kernels.xpu_fused_moe(is_int4=True)`. Upstream is building its own XPU
INT4 MoE (PRs #42032 / #41426); this branch is the interim until that lands.

---

## Environment — get this right or nothing works

### Do NOT `source setvars.sh` to *run* vLLM
The modern `torch-xpu` wheel ships a complete, self-contained SYCL runtime
(`libsycl`, the UR loader, the Level-Zero adapters — all venv pip packages,
resolved by RPATH). `source /opt/intel/oneapi/setvars.sh` prepends an oneAPI
directory to `LD_LIBRARY_PATH` and **shadows the venv's UR loader with a
mismatched system one** → SIGSEGV at XPU init, or `Backends mismatch` on the MoE
forward. `setvars.sh` is a **build-time tool only**.

Launchers should instead **scrub** oneAPI off `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '/opt/intel/oneapi' | paste -sd: -)
```

### Keep one consistent oneAPI version
If several oneAPI versions are installed, the `latest` symlink decides what
`setvars.sh` (build time) pulls in. Point it at the version consistent with the
`torch-xpu` wheel, and stop package updates from silently adding a newer one
(e.g. disable the Intel oneAPI dnf/apt repo):
```bash
ls -la /opt/intel/oneapi/compiler/latest   # one consistent version
```

### Clear a stale Triton cache after any toolchain change
Triton-XPU compiles a `spirv_utils` helper module on first use and **caches it**
in `~/.triton/cache`. If it was compiled while a different oneAPI was active, the
cached `.so` has the wrong absolute `libsycl` path frozen in and reloads forever
— repointing `latest` does **not** fix it. After any oneAPI change:
```bash
mv ~/.triton/cache ~/.triton/cache.stale   # Triton rebuilds it clean
```

---

## Install (from this branch)

```bash
git clone -b lunar-lake https://github.com/MegaStood/vllm-for-lunar-lake.git
cd vllm-for-lunar-lake

python3.12 -m venv ~/vllm-ll-venv
source ~/vllm-ll-venv/bin/activate
pip install --upgrade pip "setuptools<80" wheel setuptools_scm

# torch
pip install torch==2.11.0+xpu torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu

# vLLM, from this branch.  oneAPI is needed ONLY for this build step;
# make sure `latest` points at the consistent version first (see Environment).
source /opt/intel/oneapi/setvars.sh --force
VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation

# kernels — install AFTER vLLM.  `pip install -e .` pins vllm-xpu-kernels==0.1.7
# and downgrades it, so force-install the stock 0.1.8.1 release LAST:
pip install --force-reinstall --no-deps \
  https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.8.1/vllm_xpu_kernels-0.1.8.1-cp38-abi3-manylinux_2_28_x86_64.whl
```

**Gotchas, in the order they bite:**
- **`setuptools<80`** — vLLM 0.21's `pyproject.toml` license schema breaks on newer setuptools.
- **Kernels last** — vLLM's `requirements/xpu.txt` pins `vllm-xpu-kernels==0.1.7`; `pip install -e .` will downgrade whatever you installed earlier. Install 0.1.8.1 *after* vLLM, with `--force-reinstall --no-deps`.
- **triton-xpu** — `triton-xpu 3.7.0` is **not on PyPI**; vLLM's deps pull CUDA `triton` instead. Replace it with `triton-xpu` from `intel/intel-xpu-backend-for-triton` (or copy a known-good `triton/` + `triton_xpu-3.7.0.dist-info` from a working venv). This is the one genuinely awkward step.

Smoke-verify:
```bash
python -c "import torch; print('xpu', torch.xpu.is_available())"
python -c "import vllm, vllm_xpu_kernels; print('vllm', vllm.__version__)"
```

---

## Running

Use a launcher that does **not** source `setvars.sh`. Minimal shape:

```bash
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '/opt/intel/oneapi' | paste -sd: -)
source ~/vllm-ll-venv/bin/activate
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SYCL_UR_USE_LEVEL_ZERO_V2=0          # pin the V1 L0 adapter
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

vllm serve <model> \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-num-seqs 1 \
    --max-model-len <tight> \
    --kv-cache-memory-bytes <bytes>
```

- `--enforce-eager` — `torch.compile` / XPU-Graph are not reliable here yet.
- `--kv-cache-memory-bytes` — set explicitly; skips a memory-profiling pass.
- Keep `--max-model-len` tight — an oversized window tanks prefill on the iGPU.

---

## Known issues / limitations

- **L0 / GTT host-memory leak.** After a vLLM run — even a clean shutdown — the
  Level-Zero driver leaves ~13–16 GiB of host RAM pinned; it is not released
  until reboot. Practical effect: **reboot between model swaps.**
- **Cold SYCL JIT.** The first run after a reboot recompiles SYCL kernels;
  `SYCL_CACHE_PERSISTENT=1` with a stable cache dir speeds up later runs.
- **Intel deprioritizes laptop XPU.** Upstream support for Lunar-Lake-class
  iGPUs is best-effort — expect to carry interim patches like these for a while.

---

## The week's dead ends (so nobody repeats them)

The `Backends mismatch` was **none** of these — all investigated and ruled out:

- AOT-rebuilding `vllm-xpu-kernels` — **not needed**; stock 0.1.8.1 works.
- An oneDNN context-reuse patch to the kernels — **not needed**; stock works.
- Swapping `libsycl.so` between toolchains — not the cause.
- "A v0.20/v0.21 code change broke MoE" — **false**; the vLLM code was fine.

The cause, end to end, was **oneAPI version mixing**: `setvars.sh` pulling a
second oneAPI (2026.0) onto `LD_LIBRARY_PATH`, and a stale Triton `spirv_utils`
cache compiled against it. Fix: nosetvars launchers + one consistent oneAPI +
clear the Triton cache.

**Lesson:** on an opaque XPU crash, audit the toolchain — oneAPI version,
`setvars`, JIT caches — *before* reading vLLM source.
