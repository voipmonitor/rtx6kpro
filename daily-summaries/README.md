# Daily Summaries

Automated daily summaries of the [RTX6kPRO Discord](https://discord.gg/FJye6yaWN3) community activity.

*Auto-generated daily at 08:07 UTC. Source: Discord bot → Claude → Discord + wiki.*


## March 2026

| Date | Highlights |
|------|------------|
| [2026-03-31](2026-03/2026-03-31.md) | b12x fused MoE kernel released (lukealonso): single-kernel launch replaces 7+ separate SGLang dispat |
| [2026-03-30](2026-03/2026-03-30.md) | b12x formally documented: 29-37% faster than cutlass at conc 1-8 for Qwen3.5-397B (92.4 vs 70.2 tok/ |
| [2026-03-29](2026-03/2026-03-29.md) | b12x stable; image `voipmonitor/sglang:test-cu130`, flags: `--fp4-gemm-backend b12x --moe-runner-bac |
| [2026-03-28](2026-03/2026-03-28.md) | FlashInfer 0.6.7 released with SM120 updates; CuteDSL kernels enable `--moe-runner-backend flashinfe |
| [2026-03-27](2026-03/2026-03-27.md) | GLM 5.1 announced by Zhipu/Zai.org; weights not yet released; no "Air" variant confirmed |
| [2026-03-26](2026-03/2026-03-26.md) | Phaelon's vLLM full-vocab LogProbs PR still the only accurate KLD implementation; stock Model Runner |
| [2026-03-25](2026-03/2026-03-25.md) | Google re-announced TurboQuant (ICLR 2026 poster); paper sat on arXiv ~11 months unnoticed |
| [2026-03-24](2026-03/2026-03-24.md) | LiteLLM supply-chain attack: versions 1.82.7/1.82.8 contained infostealer+C2 malware; window ~3 hour |
| [2026-03-23](2026-03/2026-03-23.md) | GLM-5: 6 cards PP = ~35 tok/s (no MTP); 8 cards MTP = ~140 tok/s |
| [2026-03-22](2026-03/2026-03-22.md) | luke announces Trough: new SM120-only inference engine, minimal TP-only serving stack to cut vLL |
| [2026-03-21](2026-03/2026-03-21.md) | FlashInfer PR #2780 merged: fixes flashinfer_cutlass race condition (not present in cudnn backend);  |
| [2026-03-20](2026-03/2026-03-20.md) | Broadcom PEX89000 PCIe Gen5 switch bug formally documented: posted-write bandwidth collapses from 37 |
| [2026-03-19](2026-03/2026-03-19.md) | b12x library released (github.com/lukealonso/b12x, `pip install b12x`): SM120-only NVFP4/MoE ker |
| [2026-03-18](2026-03/2026-03-18.md) | MiniMax M2.7 announced: self-trained through 100+ autonomous improvement rounds, 30% capability gain |
| [2026-03-17](2026-03/2026-03-17.md) | RubenD confirms SGLang + flashinfer patch + BF16 KV cache delivers dramatic speed improvement at lon |
| [2026-03-16](2026-03/2026-03-16.md) | SGLang long-context fix: `--attention-backend flashinfer` resolves 20 tok/s at 128k — triton is brok |
| [2026-03-15](2026-03/2026-03-15.md) | Custom PCIe allreduce (luke's fork) beats NCCL by 2–4x for payloads <64 KB; root's 7-GPU setup shows |
| [2026-03-14](2026-03/2026-03-14.md) | PCIe switch topology impact confirmed for orangezed: 8-GPU p2pmark shows cross-switch latency 1.30–1 |
| [2026-03-13](2026-03/2026-03-13.md) | Festr released llm-inference-bench tool: single Python script for standardized decode throughput |
| [2026-03-12](2026-03/2026-03-12.md) | NVFP4 quality concerns solidifying: Festr confirms AWQ quant beats NVFP4 on KLD tests; Unsloth dropp |
| [2026-03-11](2026-03/2026-03-11.md) | Luke's Qwen3.5-397B NVFP4 quant now works on vLLM with no special params (confirmed); also fixed vLL |
| [2026-03-10](2026-03/2026-03-10.md) | Qwen3.5-397B NVFP4 throughput on 8x RTX Pro 6000 (SGLang, TP8+EP8, MTP): Luke reports ~275 tok/s sin |
| [2026-03-09](2026-03/2026-03-09.md) | GLM-5 on 6x RTX Pro 6000 (TP2 PP3) confirmed working but only 24 tok/s single request, no MTP suppor |
| [2026-03-08](2026-03/2026-03-08.md) | Luke released p2pmark benchmark tool (github.com/lukealonso/p2pmark) measuring PCIe link score a |
| [2026-03-07](2026-03/2026-03-07.md) | GLM-5 FP4 GEMM race condition fix: FlashInfer PR #2716 merged; `--fp4-gemm-backend flashinfer_cutlas |
| [2026-03-06](2026-03/2026-03-06.md) | GLM-5 NaN crash root cause found: `flashinfer_cutlass` FP4 GEMM backend has a race condition causing |
| [2026-03-05](2026-03/2026-03-05.md) | GLM-5 deep_gemm clarification: `--moe-runner-backend deep_gemm` silently falls back to `cutlass` on  |
| [2026-03-04](2026-03/2026-03-04.md) | MTP root cause for GLM tool call failures identified: when `thinking: true` + MTP enabled, model out |
| [2026-03-03](2026-03/2026-03-03.md) | GLM-5 NVFP4 MMLU benchmark vs reference: average accuracy 0.873 (nvfp4) vs 0.877 (BF16 reference) —  |
| [2026-03-02](2026-03/2026-03-02.md) | FlashInfer SM120f support: PR #2650 merged adding `gen_fp4_quantization_sm120f_module`; key insight  |
| [2026-03-01](2026-03/2026-03-01.md) | GLM-5 NVFP4 with MTP enabled: Festr reports speed doubling — ~100 tok/s at 0 context, 32 tok/s at 10 |

## February 2026

| Date | Highlights |
|------|------------|
| [2026-03-31](2026-03/2026-03-31.md) | b12x fused MoE kernel released (lukealonso): single-kernel launch replaces 7+ separate SGLang dispat |
| [2026-02-28](2026-02/2026-02-28.md) | GLM-5 NVFP4 confirmed running on 8x RTX 6000 via SGLang (TP8, bf16 KV): 44 tok/s at 0 context, 30 to |
| [2026-02-27](2026-02/2026-02-27.md) | MiniMax M2.5 NVFP4 on 2x RTX 6000 Pro (300W) vs Full FP8 on 4x RTX 6000 Pro (300W): NVFP4 2-card ver |
| [2026-02-26](2026-02/2026-02-26.md) | Deepseek reportedly withholding latest AI model from US chipmakers including Nvidia (Reuters); commu |
| [2026-02-25](2026-02/2026-02-25.md) | Qwen3.5-397B-A17B NVFP4 working in vLLM cu130-nightly at ~73-78 tok/s on 4x RTX 6000 Pro (darkstar00 |
| [2026-02-24](2026-02/2026-02-24.md) | High-concurrency benchmark: Kimi K2.5 on 8 cards, 100 concurrent requests at 40K token context with  |
| [2026-02-23](2026-02/2026-02-23.md) | Root cause of Festr's AMD Turin slowdown found: NCCL v2.28.3 hardcodes AMD BW at 16 GB/s for all AMD |
| [2026-02-22](2026-02/2026-02-22.md) | Festr confirmed venv performance matches docker after full rebuild — performance delta was software, |
| [2026-02-21](2026-02/2026-02-21.md) | Root cause of Festr's 2-socket slowdown identified: AMD EPYC dual-CPU PCIe P2P must cross xGMI/IF in |
| [2026-02-20](2026-02/2026-02-20.md) | Grimulkan confirmed Kimi K2.5 on 16x RTX 6000 Pro (TP16, FP8 KV): ~68 tok/s single batch, up to ~170 |
| [2026-02-19](2026-02/2026-02-19.md) | Qwen3.5-397B-A17B FP8 confirmed running on 8x GPU via SGLang with MTP speculative decoding: 75-125 t |
| [2026-02-18](2026-02/2026-02-18.md) | GLM-5 on 8x GPU: only 35-36 tok/s (BF16 KV cache) vs luke's ~50 tok/s — gap being investigated |
| [2026-02-17](2026-02/2026-02-17.md) | GLM-5 NVFP4 (`lukealonso/GLM-5-NVFP4`): 440GB VRAM for weights; ~50 tok/s on 8x RTX Pro 6000; workin |
| [2026-02-16](2026-02/2026-02-16.md) | Qwen3.5 (807GB model weights) discussed — requires 8x GPU even for FP8; NVFP4 needed for 4-card depl |
| [2026-02-15](2026-02/2026-02-15.md) | GLM-4.7 FP8 launch command documented for 4x GPU: `USE_TRITON_W8A8_FP8_KERNEL=1`, `SGL_DISABLE_TP_ME |
| [2026-02-14](2026-02/2026-02-14.md) | MiniMax-M2.5 NVFP4 quant tested: INT4 variant (`mratsim/Minimax-M2.5-BF16-INT4-AWQ`) has >15% coding |
| [2026-02-13](2026-02/2026-02-13.md) | MiniMax-M2.5 confirmed running on 8x RTX Pro 6000: 70 tok/s single connection, 122 tok/s dual connec |
| [2026-02-12](2026-02/2026-02-12.md) | MiniMax-M2.1 confirmed running on 4x RTX Pro 6000 (official 8-bit weights, ~230GB); vLLM launch comm |
