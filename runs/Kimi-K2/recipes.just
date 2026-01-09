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
	--capture-range-end "stop" \
'''


# -------
# Recipes
# -------

prefill-master PMA data_parallel_size:
	{{PREFILL_VLLM_ENV}}  vllm serve {{MODEL}} \
		{{PREFILL_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-size {{data_parallel_size}}
		
prefill-worker PMA DPSR data_parallel_size:
	{{PREFILL_VLLM_ENV}}  vllm serve {{MODEL}} \
		{{PREFILL_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{data_parallel_size}} \
		--headless \
		2>&1 | tee prefill-worker-{{DPSR}}.log

decode-master DMA data_parallel_size:
	{{DECODE_VLLM_ENV}} {{NSYS_COMMAND}} -o decode_master.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee decode-master.log
		
decode-worker DMA DPSR data_parallel_size:
	{{DECODE_VLLM_ENV}} {{NSYS_COMMAND}} -o decode_worker.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--headless \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee decode-worker-{{DPSR}}.log

prefill-master-pd PMA data_parallel_size:
	{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o prefill_master.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_PD_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee prefill-node0-pd.log

prefill-master-pd-node1 PMA data_parallel_size:
	{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o prefill_node1.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_PD_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee prefill-node1-pd.log

prefill-master-pd-node2 PMA data_parallel_size:
	{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o prefill_node2.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_PD_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee prefill-node2-pd.log

prefill-worker-pd PMA DPSR data_parallel_size:
	{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o prefill_worker.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_PD_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee prefill-worker-{{DPSR}}.log

decode-master-pd DMA data_parallel_size:
	{{DECODE_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o decode_master.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_PD_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee decode-master-pd.log

decode-worker-pd DMA DPSR data_parallel_size:
	{{DECODE_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o decode_worker_{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_PD_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{data_parallel_size}} \
		2>&1 | tee decode-worker-pd-{{DPSR}}.log

# Router for 2 prefill (TP=8 each, hybrid-lb) + 1 decode (TP=8, hybrid-lb) configuration
# 4 prefill endpoints + 2 decode endpoints
router PA0M PA0W PA1M PA1W DA0 DA1:
	#!/usr/bin/env bash
	pushd /vllm-workspace/router
	RUST_LOG=warn \
	cargo run --release -- \
		--policy round_robin \
		--vllm-pd-disaggregation \
		--max-concurrent-requests 9216 \
		--prefill {{PA0M}} \
		--prefill {{PA0W}} \
		--prefill {{PA1M}} \
		--prefill {{PA1W}} \
		--decode {{DA0}} \
		--decode {{DA1}} \
		--host 0.0.0.0 \
		--port 8123 \
		--intra-node-data-parallel-size 4 \
		2>&1 | tee router.log
	popd

bench PMA="" PORT="8087" max_concurrency="8096" ISL="1024" OSL="1024":
	#!/usr/bin/env bash
	# Source utils.sh (assumes running from runs/fp4/ directory)
	source ../utils.sh
	#if [ "$nsys" = "true" ]; then
	#	start_profiler_control "{{PMA}}" "{{PORT}}"
	#fi
	#start_profiler_control "{{PMA}}" "{{PORT}}" &
	vllm bench serve \
		--model {{MODEL}} \
		--host {{PMA}} \
		--port {{PORT}} \
		--dataset-name random \
		--ignore-eos \
		--num-prompts $((5 * {{max_concurrency}})) \
		--max-concurrency $((2 * {{max_concurrency}})) \
		--random-input-len {{ISL}} \
		--random-output-len {{OSL}} \
		--ready-check-timeout-sec 0 \
		--num-skip-requests 0 \
		--trust_remote_code \
		2>&1 | tee bench.log

accuracy MA="" PORT="8087":
	lm_eval --model local-completions --tasks gsm8k --trust_remote_code --batch_size auto \
		--model_args base_url=http://{{MA}}:{{PORT}}/v1/completions,pretrained={{MODEL}},model={{MODEL}},add_bos_token=True,tokenized_requests=False,tokenizer_backend=None,num_concurrent=1024,timeout=60000,max_retries=5 \
		2>&1 | tee eval.log

kill:
	pkill -9 -f VLLM::EngineCore || echo "All processes stopped"
