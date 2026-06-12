# Launchers — vLLM 0.22 + Lunar Lake

Each script in this folder is a `vllm serve` wrapper used in development on the
MSI Claw 8 AI+ (Lunar Lake, Arc 140V iGPU, 32 GB UMA).

**These are verbatim copies from `~/bin/` on the developer's machine.** Paths
inside (model paths under `/shared/models/...`, venv at `~/vllm-v22-venv`,
SYCL cache dirs, etc.) are NOT portable. **Adapt the constants at the top of
each file to your layout** before running.

## Common pattern

Every launcher does the same five things, in order:

1. **Scrub the oneAPI environment** — unset `ONEAPI_*`, `CMPLR_*`,
   `SETVARS_COMPLETED`, and filter `/opt/intel/oneapi/*` out of
   `LD_LIBRARY_PATH`. This is the single most important step;
   skipping it causes `sycl::errc::backend_mismatch` at startup.
2. **Activate the v0.22 venv** (`~/vllm-v22-venv` here — adapt).
3. **Set per-launcher SYCL/NEO cache dirs** so v0.22 caches don't clash with
   v0.21 fork caches.
4. **Set vLLM XPU env knobs** (typically `VLLM_USE_TRITON_FLASH_ATTN`,
   `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TORCH_LLM_ALLOW_TF32=0`).
5. `exec vllm serve <model> --device xpu ...` with model-specific flags.

## What's here

| Launcher | Model | Notes |
|---|---|---|
| `vllm-nemotron-3-nano-v22` | `nvidia/Nemotron-3-Nano-30B-A3B` (AWQ from stelterlab) | recommended starting point — most-stress-tested |
| `vllm-lfm2-24b-v22` | `LiquidAI/LFM2-24B-A2B` (compressed-tensors WNA16) | flat decode across full 32K |
| `vllm-lfm2-8b-fp8-v22` | `LiquidAI/LFM2.5-8B` FP8 | reasoning model — emits `<think>` blocks |
| `vllm-gemma4-qat-v22` | `google/gemma-4-26b-a4b-it-qat-w4a16-ct` | with optional spec-decode drafter |
| `vllm-qwen3.5-35b-v22` | `cyankiwi/Qwen3.5-35B-A3B-Instruct-AWQ-INT4` | needs extra `_process_weights_xpu` tuple-shape fix (not in this branch) |
| `vllm-qwen3.5-35b-gptq-v22` | `cyankiwi/Qwen3.5-35B-A3B-Instruct-GPTQ-INT4` | sibling GPTQ variant |

## How to adapt

1. Edit the model path at the top of the file to your local cache location.
2. Edit `VENV=` (or the `source` line) to your v0.22 venv path.
3. Edit `SYCL_CACHE_DIR` / `NEO_CACHE_DIR` to writable locations.
4. Sanity-check the `--gpu-memory-utilization` and `--max-model-len` flags
   against your iGPU's UMA budget (32 GB on the Claw 8 — you'll have less
   wiggle room on 16 GB Arrow Lake-H).
5. Run `bash -n launchers/vllm-XXX-v22` to syntax-check before serving.

If `serve` fails with `backend_mismatch` or `device_id` errors, the most
common cause is that step 1 (oneAPI scrub) didn't fully take — check that
no parent shell sourced `setvars.sh` before you ran the launcher.
