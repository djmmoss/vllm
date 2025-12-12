# ==============================================================================
# vLLM FP4 & SGLang Deployment Justfile
# ==============================================================================
# Quick Start:
#   just help           # Show all available commands
#   just prefill        # Start vLLM prefill (4 GPUs)
#   just decode         # Start vLLM decode
#   just sgl            # Start SGLang (4 GPUs)
#   just bench          # Run benchmark
#
# Related code branch: https://github.com/minosfuture/vllm/tree/pd_gb200
#
# P/D Example
#
# 1P:DP2EP2 + 1D:DP8EP8
# Assuming 4 GPUs per node:
#
# # on node 1
# just prefill-off
# # on node 2
# just decode
# # on node 3
# just decode-worker
# # run router anywhere
# just router
# ==============================================================================

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MODEL := "nvidia/DeepSeek-R1-0528-FP4-v2"
HF_CACHE_HOME := "/data/numa0/ming_hf_cache/"
DECODE_MASTER := "192.168.5.50"   # Decode cluster master node IP
NSYS := ""

export HF_HOME := HF_CACHE_HOME
export FLASHINFER_CACHE_DIR := "/data/nfs01/ming/.cache/flashinfer/"

# ------------------------------------------------------------------------------
# vLLM Environment Variables
# ------------------------------------------------------------------------------

SYSTEM_ENV := '''
NVIDIA_GDRCOPY=1 \
NVSHMEM_IB_ENABLE_IBGDA=1 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=enP22p3s0f1np1 \
UCX_IB_ROCE_REACHABILITY_MODE=local_subnet \
VLLM_SKIP_P2P_CHECK=1 \
GLOO_SOCKET_IFNAME=enP22p3s0f1np1 \
NCCL_SOCKET_IFNAME=enP22p3s0f1np1 \
NCCL_CUMEM_ENABLE=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_NVLS_ENABLE=1 \
CUDA_HOME=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
PATH=/usr/local/cuda/bin:$PATH \
UCX_TLS=all \
VLLM_LOGGING_LEVEL=INFO \
VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300 \
VLLM_NIXL_SIDE_CHANNEL_HOST=`hostname -i` \
VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
VLLM_TORCH_PROFILER_DIR=./profile/ \
VLLM_USE_DEEP_GEMM=0 \
'''

COMMON_VLLM_ENV := '''
VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA \
VLLM_USE_FLASHINFER_MOE_FP4=1 \
VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
VLLM_USE_NCCL_SYMM_MEM=1 \
'''

PREFILL_VLLM_ENV := SYSTEM_ENV + COMMON_VLLM_ENV
PREFILL_PD_VLLM_ENV := PREFILL_VLLM_ENV + PD_VLLM_ENV
DECODE_VLLM_ENV := SYSTEM_ENV + COMMON_VLLM_ENV
DECODE_PD_VLLM_ENV := DECODE_VLLM_ENV + PD_VLLM_ENV

PREFILL_ENV := PREFILL_VLLM_ENV
DECODE_ENV := DECODE_VLLM_ENV

# ------------------------------------------------------------------------------
# vLLM Arguments
# ------------------------------------------------------------------------------

COMMON_VLLM_ARGS := '''
--kv-cache-dtype fp8 \
--tensor-parallel-size 1 \
--pipeline-parallel-size 1 \
--enable-expert-parallel \
--data-parallel-rpc-port 13345 \
--max-model-len 4096 \
--data-parallel-size-local 4 \
--disable-uvicorn-access-log \
--no-enable-prefix-caching \
--port 8087 \
--async-scheduling \
--disable-uvicorn-access-log \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
--trust-remote-code \
'''

PREFILL_VLLM_ARGS := COMMON_VLLM_ARGS + '''
--swap-space 16 \
--max-num-seqs 128 \
--compilation-config '{"cudagraph_mode":"NONE"}' \
--gpu-memory-utilization 0.85 \
--max-num-batched-tokens 32768 \
--compilation_config.pass_config.enable_fi_allreduce_fusion true \
--compilation_config.pass_config.enable_attn_fusion true \
--compilation_config.pass_config.enable_noop true \
--compilation_config.custom_ops+=+quant_fp8,+rms_norm \
--enable-eplb \
--eplb-config '{"window_size":"100", "step_interval":"500", "num_redundant_experts":"32", "log_balancedness":"False"}' \
'''

PREFILL_PD_VLLM_ARGS := PREFILL_VLLM_ARGS + PD_VLLM_ARGS
PREFILL_ARGS := PREFILL_VLLM_ARGS

DECODE_VLLM_ARGS := COMMON_VLLM_ARGS + '''
--all2all-backend allgather_reducescatter \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--data-parallel-hybrid-lb \
--gpu-memory-utilization 0.9 \
--stream-interval 50 \
--max-num-seqs 1024 \
--max-num-batched-tokens 16384 \
--max-cudagraph-capture-size 2048 \
'''

DECODE_PD_VLLM_ARGS := DECODE_VLLM_ARGS + PD_VLLM_ARGS
DECODE_ARGS := DECODE_VLLM_ARGS

# ==============================================================================
# vLLM Recipes
# ==============================================================================

# Start prefill (4 GPUs)
prefill:
    {{PREFILL_ENV}} \
    {{NSYS}} vllm serve {{MODEL}} \
        {{PREFILL_ARGS}} \
        --data-parallel-size 4 \
        2>&1 | tee prefill.log

# Start prefill with offloading (2 GPUs)
# check nvidia-smi topo -m for numa-GPU affinity
prefill-off NUMA="0" PORT="8000":
    {{PREFILL_ENV}} \
    {{NSYS}} \
    numactl --cpunodebind={{NUMA}} --membind={{NUMA}} \
    vllm serve {{MODEL}} \
        {{PREFILL_ARGS}} \
        --port {{PORT}} \
        --data-parallel-size 2 \
        --enforce-eager \
        --gpu-memory-utilization 0.84 \
        --offload-group-size 2 \
        --offload-num-in-group 1 \
        --offload-prefetch-step 1 \
        2>&1 | tee prefill-off.log

# Start lead decode
decode:
    {{DECODE_ENV}} vllm serve {{MODEL}} \
        {{DECODE_ARGS}} \
        --data-parallel-address `hostname -i` \
        2>&1 | tee decode.log

# Start worker decode
decode-worker DPSR="4":
    {{DECODE_ENV}} vllm serve {{MODEL}} \
        {{DECODE_ARGS}} \
        --data-parallel-address {{DECODE_MASTER}} \
        --data-parallel-start-rank={{DPSR}} \
        2>&1 | tee decode.log

# ------------------------------------------------------------------------------
# SGLang Configuration
# ------------------------------------------------------------------------------

SGL_PORT := "10000"

SGL_ENV := '''
FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1 \
GLOO_SOCKET_IFNAME=eth0 \
MC_TE_METRIC=true \
NCCL_CUMEM_ENABLE=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_SOCKET_IFNAME=eth0 \
PYTHONUNBUFFERED=1 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_LOCAL_IP_NIC=eth0 \
SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
SGLANG_TORCH_PROFILER_DIR=/data/nfs01/ming/sglang_gb200/profile \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
UNBALANCED_MODEL_LOADING_TIMEOUT_S=36000 \
'''

SGL_ARGS := '''
--attention-backend trtllm_mla \
--chunked-prefill-size 65536 \
--context-length 4224 \
--decode-log-interval 1 \
--disable-chunked-prefix-cache \
--disable-cuda-graph \
--disable-radix-cache \
--disable-shared-experts-fusion \
--enable-dp-attention \
--enable-dp-lm-head \
--eplb-algorithm deepseek \
--host 0.0.0.0 \
--kv-cache-dtype fp8_e4m3 \
--load-balance-method round_robin \
--max-running-requests 768 \
--max-total-tokens 131072 \
--mem-fraction-static 0.84 \
--moe-dense-tp-size 1 \
--moe-runner-backend flashinfer_cutlass \
--offload-mode cpu \
--quantization modelopt_fp4 \
--trust-remote-code \
--watchdog-timeout 1000000 \
'''


# ==============================================================================
# SGLang Recipes
# ==============================================================================

# Start SGLang (4 GPUs)
sgl GPU="4":
    {{SGL_ENV}} \
    numactl --cpunodebind=0 --membind=0 \
    python -m sglang.launch_server \
        {{SGL_ARGS}} \
        --dp-size {{GPU}} \
        --ep {{GPU}} \
        --max-prefill-tokens 16384 \
        --model-path {{MODEL}} \
        --port {{SGL_PORT}} \
        --tp-size {{GPU}}

# Start SGLang with offloading (2 GPUs)
sgl-off GPU="2":
    {{SGL_ENV}} \
    numactl --cpunodebind=0 --membind=0 \
    python -m sglang.launch_server \
        {{SGL_ARGS}} \
        --dp-size {{GPU}} \
        --ep {{GPU}} \
        --max-prefill-tokens 32768 \
        --model-path {{MODEL}} \
        --offload-group-size 2 \
        --offload-num-in-group 1 \
        --offload-prefetch-step 1 \
        --port {{SGL_PORT}} \
        --tp-size {{GPU}}

# ==============================================================================
# Testing & Benchmarking
# ==============================================================================

quickrun PORT="8000":
    curl http://localhost:{{PORT}}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

bench BS="4096" RATE="inf" ISL="2048" OSL="1024" PORT="8192" COMMON_PREFIX="0":
    #!/usr/bin/env bash
    while ! curl -s "http://localhost:{{PORT}}/health" >/dev/null 2>&1; do
        echo -n "."
        sleep 5
    done
    vllm bench serve \
        --common-prefix-len {{COMMON_PREFIX}} \
        --dataset-name random \
        --ignore-eos \
        --max-concurrency {{BS}} \
        --model {{MODEL}} \
        --num-prompts {{BS}} \
        --port {{PORT}} \
        --random-input-len {{ISL}} \
        --random-output-len {{OSL}} \
        --ready-check-timeout-sec 0 \
        --request-rate {{RATE}} \
        --seed $RANDOM \
        --trust_remote_code

# Uses short output (1 token) to measure prefill throughput
# Benchmark prefill performance
bench-prefill BS="1024" RATE="inf" ISL="2048" OSL="1" PORT="8000":
    just bench {{BS}} {{RATE}} {{ISL}} {{OSL}} {{PORT}}

# Uses short input (2 tokens) to measure decode throughput
# Requires prefix len configuration on vllm bench serve side
# Benchmark decode performance
bench-decode BS="4096" RATE="inf" ISL="2" OSL="1024" PORT="8000" COMMON_PREFIX="2047":
    just bench {{BS}} {{RATE}} {{ISL}} {{OSL}} {{PORT}} {{COMMON_PREFIX}}


sgl-bench:
    python3 -m sglang.bench_one_batch_server \
        --base-url http://localhost:{{SGL_PORT}} \
        --model-path {{MODEL}} \
        --batch-size 1024 \
        --input-len 2048 \
        --output-len 1 \
        --skip-warmup

eval:
    lm_eval --model local-completions --tasks gsm8k \
        --model_args model={{MODEL}},base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=32 \
        --limit 100

# ==============================================================================
# Profiling
# ==============================================================================

profile URL="localhost:8000" DUR="1":
    curl -X POST {{URL}}/start_profile && sleep {{DUR}} && curl -X POST {{URL}}/stop_profile

sgl-profile:
    curl -X POST -H 'Content-Type: application/json' \
        "http://localhost:{{SGL_PORT}}/start_profile" \
        -d '{"num_steps":4}'

plot LOG="prefill.log" OUTPUT="metrics.pdf" TIME="":
    #!/usr/bin/env bash
    if [ -n "{{TIME}}" ]; then
        ./plot_metrics.py {{LOG}} -o {{OUTPUT}} --start-time={{TIME}}
    else
        ./plot_metrics.py {{LOG}} -o {{OUTPUT}}
    fi

# ==============================================================================
# Router
# ==============================================================================

router:
    #!/usr/bin/env bash
    pushd /data/nfs01/ming/router/
    RUST_LOG=warn \
    cargo run --release -- \
        --policy consistent_hash \
        --vllm-pd-disaggregation \
        --max-concurrent-requests 8192 \
        --prefill http://192.168.5.86:8000 \
        --prefill http://192.168.5.86:8010 \
        --prefill http://192.168.5.48:8000 \
        --prefill http://192.168.5.48:8010 \
        --prefill http://192.168.5.82:8000 \
        --prefill http://192.168.5.82:8010 \
        --decode http://192.168.5.40:8000 \
        --decode http://192.168.5.50:8000 \
        --host 127.0.0.1 \
        --port 8192 \
        --intra-node-data-parallel-size 4
    popd

build-router:
    #!/usr/bin/env bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    pushd /data/nfs01/ming/router/
    cargo build --release
    popd

# ==============================================================================
# Utilities
# ==============================================================================

edit:
    nvim justfile

clean-log:
    rm -f *.log

clean-cache:
    rm -rf $HOME/.cache/vllm $HOME/.cache/flashinfer/

kill:
    pkill -9 -f VLLM::EngineCore || echo "All processes stopped"

build:
    CCACHE_NOHASHDIR="true" uv pip install --editable . --prerelease=allow \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --index-strategy unsafe-best-match -v

download:
    HF_HOME={{HF_CACHE_HOME}} hf download {{MODEL}}

help:
    @echo "================================================================================"
    @echo "vLLM FP4 & SGLang Deployment"
    @echo "================================================================================"
    @echo ""
    @echo "VLLM:"
    @echo "  just prefill           - Start prefill (4 GPUs)"
    @echo "  just prefill-off       - Start prefill with offloading (2 GPUs)"
    @echo "  just decode            - Start decode (4 GPUs)"
    @echo ""
    @echo "SGLANG:"
    @echo "  just sgl               - Start SGLang (4 GPUs)"
    @echo "  just sgl-off           - Start SGLang with offloading (2 GPUs)"
    @echo ""
    @echo "TESTING:"
    @echo "  just quickrun [PORT]   - Send test request"
    @echo "  just bench             - Run benchmark"
    @echo "  just sgl-bench         - Run SGLang benchmark"
    @echo "  just eval              - Run accuracy evaluation"
    @echo ""
    @echo "PROFILING:"
    @echo "  just profile [URL] [DUR]"
    @echo "  just sgl-profile"
    @echo "  just plot [LOG] [OUTPUT] [TIME]"
    @echo ""
    @echo "UTILITIES:"
    @echo "  just clean             - Remove log files"
    @echo "  just clean-cache       - Remove caches"
    @echo "  just kill              - Kill all vLLM processes"
    @echo "  just build             - Build vLLM"
    @echo "  just download          - Download model"
    @echo "================================================================================"
