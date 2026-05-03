# GLM-5.1 on RTX PRO 6000 Blackwell

This section tracks GLM-5.1 experiments on RTX PRO 6000 Blackwell PCIe systems.

## Reports

| Date | Report | Scope |
|---|---|---|
| 2026-05-03 | [GLM-5.1 cc32 SGLang vs vLLM runtime state](glm51-sglang-vllm-cc32-state-2026-05-03.md) | Current DCP=1 / TP=8 launch state for SGLang and vLLM at 32 concurrent decode requests, DockerHub image tags, exact launch overrides, and measured default/safe MTP throughput. |
| 2026-05-02 | [vLLM b12x NSA/MTP port: fast prefill, PCIe barriers, and upstream delta](vllm-b12x-nsa-mtp-port-2026-05-02.md) | GLM-5.1 NVFP4-MTP on vLLM with b12x sparse NSA, ModelOpt FP4, MTP, FP8 KV cache, PCIe allreduce, JIT/prefill fixes, and differences from vanilla b12x / Luke's SGLang patches. |
| 2026-04-20 | [Dense MLA vs NSA vs vLLM benchmark](compare-dense-mla-vs-nsa-benchmark-2026-04-20/README.md) | Quality and throughput comparison of SGLang dense MLA, SGLang NSA, and a vLLM reference path. |
