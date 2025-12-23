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


# -------
# Recipes
# -------

prefill-master PMA DPS:
	{{PREFILL_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-master.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-size {{DPS}} \
		2>&1 | tee prefill-master.log
		
prefill-worker PMA DPSR DPS:
	{{PREFILL_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-worker-{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{DPS}} \
		--headless \
		2>&1 | tee prefill-worker-{{DPSR}}.log

decode-master DMA DPS:
	{{DECODE_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-master.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--data-parallel-size {{DPS}} \
		2>&1 | tee decode-master.log
		
decode-worker DMA DPSR DPS:
	{{DECODE_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-worker-{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{DPS}} \
		2>&1 | tee decode-worker-{{DPSR}}.log

prefill-master-pd PMA DPS:
	{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-master.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_PD_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-size {{DPS}} \
		2>&1 | tee prefill-master-pd.log

prefill-worker-pd PMA DPSR DPS:
	{{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-worker-{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
		{{PREFILL_PD_VLLM_ARGS}} \
		--data-parallel-address {{PMA}} \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{DPS}} \
		2>&1 | tee prefill-worker-{{DPSR}}.log

decode-master-pd DMA DPS:
	{{DECODE_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-master.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_PD_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--data-parallel-size {{DPS}} \
		2>&1 | tee decode-master-pd.log

decode-worker-pd DMA DPSR DPS:
	{{DECODE_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-worker-{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
		{{DECODE_PD_VLLM_ARGS}} \
		--data-parallel-address {{DMA}} \
		--data-parallel-start-rank {{DPSR}} \
		--data-parallel-size {{DPS}} \
		2>&1 | tee decode-worker-pd-{{DPSR}}.log

proxy PMA DMA:
	python3 /scratch/toy_proxy_server.py \
		--port 8192 \
		--prefiller-hosts {{PMA}} \
		--prefiller-ports 8087 \
		--decoder-hosts {{DMA}} \
		--decoder-ports 8087 \
		2>&1 | tee proxy.log

router PA1 PA2 PA3 PA4 DA1 DA2:
	#!/usr/bin/env bash
	pushd /vllm-workspace/router
	RUST_LOG=warn \
	cargo run --release -- \
		--policy round_robin \
		--vllm-pd-disaggregation \
		--max-concurrent-requests 16384 \
		--prefill {{PA1}} \
		--prefill {{PA2}} \
		--prefill {{PA3}} \
		--prefill {{PA4}} \
		--decode {{DA1}} \
		--decode {{DA2}} \
		--host 0.0.0.0 \
		--port 8192 \
		--intra-node-data-parallel-size 4
	popd

bench PMA="" PORT="8087" MC="8096" ISL="1024" OSL="1024" COMMON_PREFIX="0" SKIP_REQS="0":
	#!/usr/bin/env bash
	# Source utils.sh (assumes running from runs/fp4/ directory)
	source ../utils.sh
	if [ "$nsys" = "true" ]; then
		start_profiler_control "{{PMA}}" "{{PORT}}"
	fi
	vllm bench serve \
		--model {{MODEL}} \
		--host {{PMA}} \
		--port {{PORT}} \
		--dataset-name random \
		--ignore-eos \
		--request-rate 100 \
		--num-prompts {{MC}} \
		--max-concurrency {{MC}} \
		--random-input-len {{ISL}} \
		--random-output-len {{OSL}} \
		--ready-check-timeout-sec 0 \
		--common-prefix-len {{COMMON_PREFIX}} \
		--num-skip-requests {{SKIP_REQS}} \
		2>&1 | tee bench.log

accuracy MA="" PORT="8087":
	lm_eval --model local-completions --tasks gsm8k --trust_remote_code --batch_size auto \
		--model_args base_url=http://{{MA}}:{{PORT}}/v1/completions,pretrained={{MODEL}},model={{MODEL}},add_bos_token=True,tokenized_requests=False,tokenizer_backend=None,num_concurrent=1024,timeout=60000,max_retries=5 \
		2>&1 | tee eval.log

kill:
	pkill -9 -f VLLM::EngineCore || echo "All processes stopped"
