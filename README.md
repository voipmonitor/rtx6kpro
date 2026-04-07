# RTX 6000 Pro Wiki — Running Large LLMs on PCIe GPUs

Community-sourced knowledge base for running large language models (Qwen3.5-397B, MiniMax M2.5, Kimi-K2.5, GLM-5) on NVIDIA RTX 6000 Pro (Blackwell, SM120) GPUs in 2×, 4×, and 8× PCIe configurations **without NVLink**.

> Synthesized from ~5,000 [Discord](https://discord.gg/pYCvaQTf) messages, 300+ screenshots, and months of community experimentation.

## Quick Links

### Models

| Model | Params | Active | Min GPUs | Best Decode | Page |
|-------|--------|--------|----------|-------------|------|
| [Qwen3.5-397B](models/qwen35-397b.md) | 397B MoE | 17B | 4× | 350 tok/s (8×, SGLang) | [→](models/qwen35-397b.md) |
| [Qwen3.5-27B/122B](models/qwen35-27b.md) | 27B–122B | — | 1× | — | [→](models/qwen35-27b.md) |
| [MiniMax M2.5](models/minimax-m25.md) | 456B MoE | — | 2× | 85-89 tok/s (NVFP4) | [→](models/minimax-m25.md) |
| [Kimi-K2.5](models/kimi-k25.md) | 530B MoE | — | 8× | 101 tok/s (PCIe switch) | [→](models/kimi-k25.md) |
| [GLM-5](models/glm5.md) | 744B MoE | 40B | 8× | 105 tok/s (MTP) | [→](models/glm5.md) |

### Hardware & Topology
- [PCIe Topology](hardware/topology.md) — Switches, Turin vs Genoa, NUMA
- [PCIe Bandwidth](hardware/pcie-bandwidth.md) — P2P measurements, BAR1, latency
- [GPU Configurations](hardware/gpu-configs.md) — 4×/8× builds, VRAM, power, rigs
- [ASUS ESC8000A-E13P + Broadcom Switches](hardware/asus-esc8000a-e13p-broadcom-switches.md) — Topology, ACS disable, P2P proof, benchmarks
- [ASRock WRX90 + 3× c-payne Switches (hierarchy)](hardware/wrx90-cpayne-microchip-switches.md) — Root switch, uniform BW, no collapse bug
- [ASRock WRX90 + 2× c-payne Switches (flat)](hardware/wrx90-cpayne-2switch-flat.md) — Flat topology, CPU-routed cross-switch, comparison
- [Cheap dual Epyc Genoa with 4/24 channels, direct attach ](hardware/epyc9124-direct-attach.md) — PCIe benchmarks and GLM-5 bench comparison to switch-hierarchy


### Inference Engines
- [vLLM](inference-engines/vllm.md) — Config, MTP, model-specific commands
- [SGLang](inference-engines/sglang.md) — Config, DCP, MOE backends
- [FlashInfer](inference-engines/flashinfer.md) — CUTLASS, SM120, bug fixes

### Optimization
- [NCCL Tuning](optimization/nccl-tuning.md) — Env vars, P2P levels, graph XML fix, tuner plugin
- [PCIe Oneshot AllReduce](optimization/pcie-oneshot-allreduce.md) — 5–11% faster decode, setup guide, benchmarks
- [NVFP4 Quantization](optimization/nvfp4-quantization.md) — Setup, calibration, models
- [Speculative Decoding](optimization/speculative-decoding.md) — MTP configs, EAGLE
- [Docker Images](optimization/docker-images.md) — Images, compose, custom builds
- [I/O Tuning (md RAID5)](optimization/io-tuning.md) — stripe_cache, group_idle, Docker overlay2

### Community
- [Daily Summaries](daily-summaries/) — Auto-generated daily digests of Discord activity

### Results & Troubleshooting
- [Benchmark Results](benchmarks/results.md) — Consolidated tables across all models
- [KLD Evaluation](benchmarks/kld-evaluation.md) — Quantization quality (KL divergence vs FP8 reference)
- [Common Issues](troubleshooting/common-issues.md) — Errors + fixes

## Key Findings

1. **MTP=2 is the sweet spot** — +51-72% throughput across all models, MTP>3 unstable
2. **NCCL graph XML fix is critical on AMD Turin** — 1.5-1.9× speedup by correcting hardcoded 16 GB/s bandwidth
3. **PCIe switches dramatically help single-batch latency** — 101 vs 60 tok/s for Kimi K2.5
4. **BF16 KV cache mandatory on SM120 for GLM-5** — FP8 produces garbled output
5. **SGLang is the only option for GLM-5** — vLLM lacks SM120-compatible MLA+sparse attention backend
6. **NVFP4 is native to SM120** — 2× decode speedup over FP8 for supported models
7. **DCP is essential for Kimi K2.5 long context** — Without it, 200K context drops to <10 tok/s

## Hardware Overview

All results are on **NVIDIA RTX PRO 6000** (Blackwell GB202, SM120):
- 96 GB GDDR7 per GPU (768 GB total for 8×)
- PCIe 5.0 x16 (~64 GB/s per direction)
- No NVLink — all inter-GPU communication via PCIe
- Typical configs: AMD EPYC Turin/Genoa, 4× or 8× GPUs

## Contributing

This wiki is synthesized from Discord discussions. If you have corrections, additional benchmarks, or new configurations, please open an issue or PR.

---

*Generated March 2026. Data sourced from community Discord server.*
