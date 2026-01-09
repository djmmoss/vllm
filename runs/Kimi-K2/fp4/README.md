# K2 Benchmarking with Prefill-Decode Disaggregation

This directory contains scripts for running K2 (Kimi-K2-Thinking-NVFP4) benchmarks with disaggregated prefill and decode instances on GB200 clusters using Hybrid Load Balancing.

## Fixed Configuration

**2 Prefill Instances (TP=8 each) + 1 Decode Instance (TP=8)**

- **Prefill**: 2 instances × TP=8 (4 nodes total: 2 nodes per instance)
- **Decode**: 1 instance × TP=8 (2 nodes)
- **Router**: 1 node for load balancing
- **Total**: 7 nodes, 24 GPUs

## Quick Start

```bash
sbatch pd.sh
```
## Architecture

### Node Allocation

```
Rank 0: Prefill 0 master (TP=8, GPUs 0-3)
Rank 1: Prefill 0 worker (TP=8, GPUs 4-7)
Rank 2: Prefill 1 master (TP=8, GPUs 0-3)
Rank 3: Prefill 1 worker (TP=8, GPUs 4-7)
Rank 4: Decode master (TP=8, GPUs 0-3)
Rank 5: Decode worker (TP=8, GPUs 4-7)
Rank 6: Router + Benchmark
```

### Hybrid Load Balancing

The router distributes requests across **6 endpoints**:

**Prefill Endpoints (4):**
- Prefill 0 master: http://node0:8087
- Prefill 0 worker: http://node1:8087
- Prefill 1 master: http://node2:8087
- Prefill 1 worker: http://node3:8087

**Decode Endpoints (2):**
- Decode master: http://node4:8087
- Decode worker: http://node5:8087

All worker nodes run in **non-headless mode** to support hybrid load balancing, allowing the router to balance requests across all available endpoints.

## Output

The script will:
1. Submit a SLURM job requesting 7 nodes
2. Create a timestamped log directory (e.g., `logs_20260105_143022/`)
3. Generate logs for each service:
   - `prefill-node0-pd.log` - Prefill 0 master
   - `prefill-worker-4.log` - Prefill 0 worker  
   - `prefill-node1-pd.log` - Prefill 1 master
   - `prefill-worker-4.log` - Prefill 1 worker
   - `decode-master-pd.log` - Decode master
   - `decode-worker-pd-4.log` - Decode worker
   - `router.log` - Router service
   - `bench.log` - Benchmark results

## Files

- `submit.sh` - Main submission script
- `disagg_bench.slurm` - SLURM batch script
- `pd-worker.sh` - Worker script executed on each node
- `justfile` - Task runner configuration (uses ../recipes.just)
