MODEL := "nvidia/DeepSeek-R1-0528-FP4-v2"
HF_CACHE_HOME := "/data/numa0/ming_hf_cache/"
export HF_HOME := HF_CACHE_HOME
export FLASHINFER_CACHE_DIR := "/data/nfs01/ming/.cache/flashinfer/"
PREC := "fp4"

# ---
# ENV
# ---

#NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=enP22p3s0f1np1 \
#UCX_IB_ROCE_REACHABILITY_MODE=local_subnet \
#GLOO_SOCKET_IFNAME=enP22p3s0f1np1 \
#NCCL_SOCKET_IFNAME=enP22p3s0f1np1 \
SYSTEM_ENV := '''NVIDIA_GDRCOPY=1 \
NVSHMEM_IB_ENABLE_IBGDA=1 \
VLLM_SKIP_P2P_CHECK=1 \
NCCL_CUMEM_ENABLE=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_NVLS_ENABLE=1 '''


COMMON_VLLM_ENV := '''\
VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA \
VLLM_USE_FLASHINFER_MOE_FP4=1 \
VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 \
VLLM_USE_NCCL_SYMM_MEM=1 '''


#VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random \
PREFILL_VLLM_ENV := SYSTEM_ENV + COMMON_VLLM_ENV + '''\
VLLM_FLASHINFER_MOE_BACKEND=latency \
VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0 \
VLLM_ENABLE_MOE_DP_CHUNK=0 \
'''

PREFILL_PD_VLLM_ENV := PREFILL_VLLM_ENV + PD_VLLM_ENV

DECODE_VLLM_ENV := SYSTEM_ENV + COMMON_VLLM_ENV + '''\
VLLM_DEEPEP_LOW_LATENCY_ALLOW_NVLINK=1 \
VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1 \
VLLM_DEEPEP_BUFFER_SIZE_MB=0 \
VLLM_EP_USE_SBO=1 \
VLLM_MOE_DP_CHUNK_SIZE=1024 \
VLLM_DEEPEPLL_NVFP4_DISPATCH=1 \
VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0 \
VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=2048 \
VLLM_FLASHINFER_MOE_BACKEND=masked_gemm '''

DECODE_PD_VLLM_ENV := DECODE_VLLM_ENV + PD_VLLM_ENV

# ---------
# vLLM Args
# ---------

COMMON_VLLM_ARGS := '''
--kv-cache-dtype fp8 \
--tensor-parallel-size 1 \
--pipeline-parallel-size 1 \
--enable-expert-parallel \
--data-parallel-rpc-port 13345 \
--max-model-len 4096 \
--data-parallel-size-local 4 \
--disable-uvicorn-access-log \
--port 8000 \
--async-scheduling '''

#--enable-eplb \
#--eplb-config '{"window_size":"100", "step_interval":"500", "num_redundant_experts":"32", "log_balancedness":"False"}' \
#--compilation_config.pass_config.enable_fi_allreduce_fusion true \
#--compilation_config.pass_config.enable_attn_fusion true \
#--compilation_config.pass_config.enable_noop true \
#--compilation_config.custom_ops+=+quant_fp8,+rms_norm \
PREFILL_VLLM_ARGS := COMMON_VLLM_ARGS + '''\
--no-enable-prefix-caching \
--swap-space 16 \
--max-num-seqs 64 \
--gpu-memory-utilization 0.85 \
--max-num-batched-tokens 65536 \
'''

PREFILL_PD_VLLM_ARGS := PREFILL_VLLM_ARGS + PD_VLLM_ARGS

DECODE_VLLM_ARGS := COMMON_VLLM_ARGS + '''\
--all2all-backend deepep_low_latency \
--data-parallel-hybrid-lb \
--stream-interval 50 \
--max-num-seqs 1024 \
--max-num-batched-tokens 2048 \
--compilation_config.cudagraph_mode=FULL_DECODE_ONLY \
--compilation_config.custom_ops+=+rms_norm,+rotary_embedding \
--max-cudagraph-capture-size 1024 '''

DECODE_PD_VLLM_ARGS := DECODE_VLLM_ARGS + PD_VLLM_ARGS

# ---
# PD
# ---

PD_VLLM_ENV := '''\
VLLM_NIXL_SIDE_CHANNEL_HOST=`hostname -i` \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300 '''

PD_VLLM_ARGS := '''\
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_load_failure_policy":"fail"}' '''

# ----
# NSYS
# ----

NSYS_COMMAND := '''\
VLLM_TORCH_CUDA_PROFILE=1 \
VLLM_TORCH_PROFILER_USE_GZIP=0 \
VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL=0 \
VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
nsys profile \
--trace=nvtx,cuda \
--force-overwrite true \
--gpu-metrics-devices=all \
--trace-fork-before-exec=true \
--cuda-graph-trace=node \
--capture-range=cudaProfilerApi \
--capture-range-end "repeat" \
'''

#{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-master.nsys-rep vllm serve {{MODEL}} \
prefill-master-pd PMA DPS:
  {{PREFILL_PD_VLLM_ENV}} VLLM_TORCH_PROFILER_DIR=./profile/ \
    vllm serve {{MODEL}} \
    {{PREFILL_PD_VLLM_ARGS}} \
    --data-parallel-address {{PMA}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee prefill-master-pd.log

#{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-master.nsys-rep \
prefill-off PMA DPS NUMA="0" :
  {{PREFILL_PD_VLLM_ENV}} VLLM_TORCH_PROFILER_DIR=./profile/ \
    numactl --cpunodebind={{NUMA}} --membind={{NUMA}} \
    vllm serve {{MODEL}} \
    {{PREFILL_PD_VLLM_ARGS}} \
    --data-parallel-address {{PMA}} \
    --data-parallel-size {{DPS}} \
    --data-parallel-size-local {{DPS}} \
    --enforce-eager \
    --offload-group-size 2 \
    --offload-num-in-group 1 \
    --offload-prefetch-step 1 \
    2>&1 | tee prefill-master-pd.log

#{{DECODE_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-master.nsys-rep \
#{{DECODE_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-worker-{{DPSR}}.nsys-rep \
decode-master-pd DMA:
    {{DECODE_PD_VLLM_ENV}} \
    VLLM_TORCH_PROFILER_DIR=./profile/ \
    vllm serve {{MODEL}} \
    {{DECODE_PD_VLLM_ARGS}} \
    --data-parallel-address {{DMA}} \
    --data-parallel-size 8 \
    2>&1 | tee decode-master-pd.log

decode-worker-pd DMA DPSR:
    {{DECODE_PD_VLLM_ENV}} \
    VLLM_TORCH_PROFILER_DIR=./profile/ \
    vllm serve {{MODEL}} \
    {{DECODE_PD_VLLM_ARGS}} \
    --data-parallel-address {{DMA}} \
    --data-parallel-start-rank {{DPSR}} \
    --data-parallel-size 8 \
    2>&1 | tee decode-worker-pd-{{DPSR}}.log

wait PORT="8000":
  #!/usr/bin/env bash
  while ! curl -s "http://localhost:{{PORT}}/health" >/dev/null 2>&1; do
  echo -n "."
  sleep 5
  done

bench BS="4096" RATE="inf" ISL="2048" OSL="1024" PORT="8192" COMMON_PREFIX="0":
  #!/usr/bin/env bash
  just wait {{PORT}}
  vllm bench serve \
    --random-prefix-len {{COMMON_PREFIX}} \
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

eval PORT="8000":
  just wait {{PORT}}
  lm_eval --model local-completions --tasks gsm8k \
    --model_args model={{MODEL}},base_url=http://127.0.0.1:{{PORT}}/v1/completions,num_concurrent=32 \
    --limit 100

eval-full PORT="8000":
  just wait
  lm_eval --model local-completions --tasks gsm8k \
    --model_args model={{MODEL}},base_url=http://127.0.0.1:{{PORT}}/v1/completions,num_concurrent=256

download:
  HF_HOME={{HF_CACHE_HOME}} hf download {{MODEL}}

router:
  #!/usr/bin/env bash
  pushd /data/nfs01/ming/router/
  RUST_LOG=warn \
    cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --max-concurrent-requests 8192 \
    --prefill http://192.168.5.82:8000 \
    --prefill http://192.168.5.82:8010 \
    --prefill http://192.168.5.98:8000 \
    --prefill http://192.168.5.98:8010 \
    --decode http://192.168.5.107:8000 \
    --decode http://192.168.5.97:8000 \
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

edit:
  nvim justfile

quickrun PORT="8000":
    curl http://localhost:{{PORT}}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

profile URL="8000" DUR="1":
    curl -X POST "http://localhost:{{URL}}/start_profile" && \
        sleep {{DUR}} && \
        curl -X POST "http://localhost:{{URL}}/stop_profile"
