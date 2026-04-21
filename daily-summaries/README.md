# Daily Summaries

Automated daily summaries of the [RTX6kPRO Discord](https://discord.gg/FJye6yaWN3) community activity.
Each highlight links directly to the relevant Discord message.

*Auto-generated daily at 08:07 UTC. Source: Discord bot → Claude → Discord + wiki.*


## March 2026

| Date | Highlights |
|------|------------|
| [2026-04-21](2026-04/2026-04-21.md) | Kimi K2.6 released (1.1T params, same arch as K2.5, vision included): SWE-Bench Pro 58.6%, HLE beati |
| [2026-04-20](2026-04/2026-04-20.md) | GLM-5.1 vLLM OOM bug fixed — new Docker `voipmonitor/vllm:glm51-tp8-nodcp-mtp3-tritondraft-b12x095 |
| [2026-04-19](2026-04/2026-04-19.md) | PCIe P2P allreduce config for ~10% throughput gain: Force-enable P2P via modprobe `options nvidia NV |
| [2026-04-18](2026-04/2026-04-18.md) | EXL3 benchmark breakthrough: mratsim's Qwen3.5-397B-A17B EXL3 quant achieves **1500 pp/s and 50+ tg/ |
| [2026-04-17](2026-04/2026-04-17.md) | Qwen3.6-35B-A3B released (MoE); community upset the poll-winner 27B wasn't chosen first — speculat |
| [2026-04-16](2026-04/2026-04-16.md) | b12x FP4 GEMM kernel merged into FlashInfer ([jump](https://discord.com/channels/1466898002793857221 |
| [2026-04-15](2026-04/2026-04-15.md) | Introspective Diffusion (I-DLM-8B) matches AR model quality, beats LLaDA-2.1-mini (16B) by +26 AIME- |
| [2026-04-14](2026-04/2026-04-14.md) | GLM 5.1 with native NSA/DSA attention now working on TP=8 with NVFP4 weights + FP8 KV cache via cust |
| [2026-04-13](2026-04/2026-04-13.md) | MiniMax M2.7 NVFP4 quant released by luke, then updated mid-day incorporating Jon's calibration data |
| [2026-04-12](2026-04/2026-04-12.md) | MiniMax M2.7 dropped (~1am) with same architecture as M2.5 — just a weight update. luke confirmed  |
| [2026-04-11](2026-04/2026-04-11.md) | vLLM MTP reaches sglang parity after a week of work by Festr; key insight was using sglang's eager-m |
| [2026-04-10](2026-04/2026-04-10.md) | GLM-5.1 NVFP4 quant live on 8x RTX Pro 6000: 131 tok/s with MTP, 95 tok/s via flashinfer cutlass. Bu |
| [2026-04-09](2026-04/2026-04-09.md) | GLM-5.1 NVFP4 upload by luke — 52 tok/s single request on 8x RTX 6000 PRO (NVFP4, b12x, sglang), t |
| [2026-04-08](2026-04/2026-04-08.md) | b12x hits 220 t/s single-user decode on 2x RTX PRO 6000; 198 t/s on Qwen3.5-122B vs 131 t/s baseline |
| [2026-04-07](2026-04/2026-04-07.md) | GLM-5 NVFP4 on 8×RTX PRO 6000: SGLang+MTP leads at 99 tok/s single-user (0 ctx), 249.8 tok/s at C=4 |
| [2026-04-06](2026-04/2026-04-06.md) | Qwen3.5 397B on 4×RTX PRO 6000: sglang record 108 tok/s (no MTP) / 180 tok/s (with MTP); vLLM now 9 |
| [2026-04-05](2026-04/2026-04-05.md) | vLLM 0.19 MoE TP regression identified and fixed — PR vllm-project/vllm#38990 awaiting merge; bran |
| [2026-04-04](2026-04/2026-04-04.md) | Qwen3.5-397B NVFP4 benchmarks on 4× RTX PRO 6000 (sglang b12x 0.7.2): MTP on = 180 tok/s single-use |
| [2026-04-03](2026-04/2026-04-03.md) | b12x 0.7.1 fixes OOM and >8 concurrency crashes on SM120; attention backend now launches for all bat |
| [2026-04-02](2026-04/2026-04-02.md) | b12x v0.7.0 released: fixes OOM on prefill, adds attention backend support (bs=1 only for now); new  |
| [2026-04-01](2026-04/2026-04-01.md) | PCIe oneshot AllReduce + fusion gives +7-8% decode throughput on SM120 across all tested models (Qwe |
| [2026-03-31](2026-03/2026-03-31.md) | b12x fused MoE kernel launched for SM120: 29–37% faster than cutlass at conc 1–8 on Qwen3.5-397B |
| [2026-03-30](2026-03/2026-03-30.md) | b12x benchmark (4x RTX 6000, TP4): 1.32x vs cutlass at conc=1-8; cutlass wins above conc=16; vLLM pe |
| [2026-03-29](2026-03/2026-03-29.md) | SGLang PR #21601: NVFP4 KV cache for Blackwell — ~2x memory vs FP8, no accuracy loss on GSM8K |
| [2026-03-28](2026-03/2026-03-28.md) | FlashInfer CuteDSL NVFP4 MoE backend landed (PR #2838); available in SGLang as `--moe-runner-backend |
| [2026-03-27](2026-03/2026-03-27.md) | GLM 5.1 announced by z.ai; no open weights yet; hosted API tested by Unoid (96 tool calls/2 min, "fe |
| [2026-03-26](2026-03/2026-03-26.md) | vLLM ModelRunner v2 still Top-K only logprobs — 1% accuracy miss; Phaelon's fork remains the only fu |
| [2026-03-25](2026-03/2026-03-25.md) | TurboQuant channel created; Google Research re-announced RaBitQ KV cache compression (ICLR 2026 post |
| [2026-03-24](2026-03/2026-03-24.md) | Festr publishes Dockerfiles to GitHub: `voipmonitor/blackwell-llm-docker` — community repo for SM120 |
| [2026-03-23](2026-03/2026-03-23.md) | GLM-5 hardware requirements: 6 cards + PP = no MTP, 35 tok/s; 8 cards + MTP = 140 tok/s; single node |
| [2026-03-22](2026-03/2026-03-22.md) | MiniMax M2.7 confirmed NOT releasing open weights; community disappointment; M2.5 remains the open o |
| [2026-03-21](2026-03/2026-03-21.md) | c-payne.com (Chris) joins: gen5 100-lane PCIe switches in stock; gen6 160-lane planned — Microchip c |
| [2026-03-20](2026-03/2026-03-20.md) | b12x SM120 MoE/FP4 kernels first Docker release: `voipmonitor/sglang:test-cu130`, 168 tok/s bench, 2 |
| [2026-03-19](2026-03/2026-03-19.md) | M2.7 weights still not out; M3 may go closed-source; GLM dropped AIR license, Qwen AI lead forced ou |
| [2026-03-18](2026-03/2026-03-18.md) | MiniMax M2.7 announced; weights expected Thu-Fri; worry about Chinese models closing source |
| [2026-03-17](2026-03/2026-03-17.md) | Flashinfer attention patch in Festr's docker (`voipmonitor/llm-pytorch-blackwell:nightly`) — sglang  |
| [2026-03-16](2026-03/2026-03-16.md) | `--attention-backend triton` causes 20 tok/s at 135k context on Qwen3.5; `flashinfer` fixes it but n |
| [2026-03-15](2026-03/2026-03-15.md) | brandonmusic's flashinfer K=64 patch confirmed by multiple independent testers to produce zero perfo |
| [2026-03-14](2026-03/2026-03-14.md) | Custom allreduce: 3.3x faster than NCCL at 256B–1KB (8 GPUs); Turin shows 3.8–5.8x at small sizes |
| [2026-03-13](2026-03/2026-03-13.md) | Festr's `llm-inference-bench`: single Python script for standardized LLM throughput testing |
| [2026-03-12](2026-03/2026-03-12.md) | KLD: AWQ lower divergence than NVFP4 for Qwen3.5; Unsloth dropped NVFP4 — poor quality and not faste |
| [2026-03-11](2026-03/2026-03-11.md) | vLLM: Qwen3.5 NVFP4 TP4+EP4+MTP3 — 150+ tok/s single, 1773 tok/s at 64 concurrent |
| [2026-03-10](2026-03/2026-03-10.md) | Kimi K2.5 EAGLE3 draft models on HF (`AQ-MedAI/Kimi-K25-eagle3`); vLLM PR #35966 adds support |
| [2026-03-09](2026-03/2026-03-09.md) | Bot posted comprehensive GLM-5 wiki pages: benchmark results, architecture overview, hardware requir |
| [2026-03-08](2026-03/2026-03-08.md) | Luke's custom PCIe allreduce beats NCCL ~2x at small batch; ~300 tok/s on Qwen3.5 NVFP4 8 GPUs |
| [2026-03-07](2026-03/2026-03-07.md) | PinchBench launched at pinchbench.com: agentic coding benchmark; GLM-5 and MiniMax M2.5 both under 4 |
| [2026-03-06](2026-03/2026-03-06.md) | GLM-5 crash: `flashinfer_cutlass` fp4-gemm triggers NaN; fix: `SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_E |
| [2026-03-05](2026-03/2026-03-05.md) | `--moe-runner-backend deep_gemm` boosts GLM-5 EAGLE by 20–30 tok/s; actually falls back to `cutlass` |
| [2026-03-04](2026-03/2026-03-04.md) | GLM-5 crash narrowed: `--fp4-gemm-backend flashinfer_cutlass` causes NaN/assertion failures; `flashi |
| [2026-03-03](2026-03/2026-03-03.md) | MTP changes model behavior: enabling MTP on Qwen3.5-NVFP4 activates different MoE experts; PR #35936 |
| [2026-03-02](2026-03/2026-03-02.md) | FlashInfer PRs #2460 and #2650: NVFP4 sm120 improvements; marginal gain over compiled vLLM (65 vs 67 |
| [2026-03-01](2026-03/2026-03-01.md) | Festr releases `claude-relay`: routes Claude Code to local models with cache sanitization |

## February 2026

| Date | Highlights |
|------|------------|
| [2026-04-21](2026-04/2026-04-21.md) | Kimi K2.6 released (1.1T params, same arch as K2.5, vision included): SWE-Bench Pro 58.6%, HLE beati |
| [2026-04-20](2026-04/2026-04-20.md) | GLM-5.1 vLLM OOM bug fixed — new Docker `voipmonitor/vllm:glm51-tp8-nodcp-mtp3-tritondraft-b12x095 |
| [2026-04-19](2026-04/2026-04-19.md) | PCIe P2P allreduce config for ~10% throughput gain: Force-enable P2P via modprobe `options nvidia NV |
| [2026-04-18](2026-04/2026-04-18.md) | EXL3 benchmark breakthrough: mratsim's Qwen3.5-397B-A17B EXL3 quant achieves **1500 pp/s and 50+ tg/ |
| [2026-04-17](2026-04/2026-04-17.md) | Qwen3.6-35B-A3B released (MoE); community upset the poll-winner 27B wasn't chosen first — speculat |
| [2026-04-16](2026-04/2026-04-16.md) | b12x FP4 GEMM kernel merged into FlashInfer ([jump](https://discord.com/channels/1466898002793857221 |
| [2026-04-15](2026-04/2026-04-15.md) | Introspective Diffusion (I-DLM-8B) matches AR model quality, beats LLaDA-2.1-mini (16B) by +26 AIME- |
| [2026-04-14](2026-04/2026-04-14.md) | GLM 5.1 with native NSA/DSA attention now working on TP=8 with NVFP4 weights + FP8 KV cache via cust |
| [2026-04-13](2026-04/2026-04-13.md) | MiniMax M2.7 NVFP4 quant released by luke, then updated mid-day incorporating Jon's calibration data |
| [2026-04-12](2026-04/2026-04-12.md) | MiniMax M2.7 dropped (~1am) with same architecture as M2.5 — just a weight update. luke confirmed  |
| [2026-04-11](2026-04/2026-04-11.md) | vLLM MTP reaches sglang parity after a week of work by Festr; key insight was using sglang's eager-m |
| [2026-04-10](2026-04/2026-04-10.md) | GLM-5.1 NVFP4 quant live on 8x RTX Pro 6000: 131 tok/s with MTP, 95 tok/s via flashinfer cutlass. Bu |
| [2026-04-09](2026-04/2026-04-09.md) | GLM-5.1 NVFP4 upload by luke — 52 tok/s single request on 8x RTX 6000 PRO (NVFP4, b12x, sglang), t |
| [2026-04-08](2026-04/2026-04-08.md) | b12x hits 220 t/s single-user decode on 2x RTX PRO 6000; 198 t/s on Qwen3.5-122B vs 131 t/s baseline |
| [2026-04-07](2026-04/2026-04-07.md) | GLM-5 NVFP4 on 8×RTX PRO 6000: SGLang+MTP leads at 99 tok/s single-user (0 ctx), 249.8 tok/s at C=4 |
| [2026-04-06](2026-04/2026-04-06.md) | Qwen3.5 397B on 4×RTX PRO 6000: sglang record 108 tok/s (no MTP) / 180 tok/s (with MTP); vLLM now 9 |
| [2026-04-05](2026-04/2026-04-05.md) | vLLM 0.19 MoE TP regression identified and fixed — PR vllm-project/vllm#38990 awaiting merge; bran |
| [2026-04-04](2026-04/2026-04-04.md) | Qwen3.5-397B NVFP4 benchmarks on 4× RTX PRO 6000 (sglang b12x 0.7.2): MTP on = 180 tok/s single-use |
| [2026-04-03](2026-04/2026-04-03.md) | b12x 0.7.1 fixes OOM and >8 concurrency crashes on SM120; attention backend now launches for all bat |
| [2026-04-02](2026-04/2026-04-02.md) | b12x v0.7.0 released: fixes OOM on prefill, adds attention backend support (bs=1 only for now); new  |
| [2026-04-01](2026-04/2026-04-01.md) | PCIe oneshot AllReduce + fusion gives +7-8% decode throughput on SM120 across all tested models (Qwe |
| [2026-02-28](2026-02/2026-02-28.md) | GLM-5 NVFP4 on 8 cards: 44 tok/s @ 0 ctx, 30 @ 150k; 4000 tok/s prefill at 400W |
| [2026-02-27](2026-02/2026-02-27.md) | Kimi K2.5 decode on Turin (8 cards, DCP=8): 65 tok/s @ 0 ctx, 36 @ 100k, 27 @ 200k |
| [2026-02-26](2026-02/2026-02-26.md) | Power benchmark: 300W→500W gives up to 30% more throughput at 64 concurrency for MiniMax M2.5 NVFP4 |
| [2026-02-25](2026-02/2026-02-25.md) | vLLM PR #34424: FP8 GEMM sm120 optimizations with smaller M-dimension kernels |
| [2026-02-24](2026-02/2026-02-24.md) | NCCL tuning on AMD XGMI: `NCCL_P2P_DISABLE=1` alone insufficient; needs full combo: `NCCL_MIN_NCHANN |
| [2026-02-23](2026-02/2026-02-23.md) | NCCL v2.28.3 bug: hardcodes AMD_BW=16 GB/s for all AMD CPUs — 12-16× underestimate for Turin xGMI3;  |
| [2026-02-22](2026-02/2026-02-22.md) | `NCCL_P2P_LEVEL=SYS` env vars close speed gap between 2-CPU and switch setups: 70 tok/s without DCP  |
| [2026-02-21](2026-02/2026-02-21.md) | Root cause of Kimi K2.5 speed gap on 2-CPU AMD system: PCIe P2P traffic must cross CPU SMP interconn |
| [2026-02-20](2026-02/2026-02-20.md) | 16-GPU Kimi K2.5 (TP16) achieves 40 tok/s single-prompt and ~1400 tok/s aggregate; TP16 beats PP2+TP |
| [2026-02-19](2026-02/2026-02-19.md) | Qwen 3.5-397B FP8 running on 8×RTX with SGLang: 75-125 tok/s, MTP speculative decoding confirmed wor |
| [2026-02-18](2026-02/2026-02-18.md) | GLM-5 NVFP4 running but only 35-36 tok/s; expected ~50 tok/s from luke's setup — investigating discr |
| [2026-02-17](2026-02/2026-02-17.md) | SGLang FP8 KV cache bug: scales applied on write but not un-applied on read — gibberish on cache reu |
| [2026-02-16](2026-02/2026-02-16.md) | Qwen 3.5 released at 807 GB weights — too large for FP8 on 8×RTX; waiting for official FP8/NVFP4 qua |
| [2026-02-15](2026-02/2026-02-15.md) | GLM 4.7 FP8 SGLang launch command shared with triton FP8 kernel, 4-GPU TP, 200K context |
| [2026-02-14](2026-02/2026-02-14.md) | GLM-5 unsupported on SM120 entirely in both vLLM and SGLang — NVFP4 the only future path |
| [2026-02-13](2026-02/2026-02-13.md) | MiniMax M2.5 FP8 confirmed running on 8x RTX 6000 Pro: 70 tok/s single, 122 tok/s dual connection, 7 |
| [2026-02-12](2026-02/2026-02-12.md) | MiniMax M2.1 working on 4x RTX 6000 Pro with official vLLM FP8 (8-bit), 96G×4 supports ~400K KV cach |
