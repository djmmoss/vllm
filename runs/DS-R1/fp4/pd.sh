#!/bin/bash

# User-specific paths
export USER_LUSTRE_DIR="/mnt/lustre/$USER"
export ENROOT_CONFIG_PATH="$USER_LUSTRE_DIR/.config/enroot"
export CONTAINER_IMAGE="$USER_LUSTRE_DIR/images/minosfuture+vllm-openai+aarch64-7434b20cf.sqsh"
export MODEL_DIR="$USER_LUSTRE_DIR/models"
export VLLM_DIR="$USER_LUSTRE_DIR/vllm"

echo $SLURM_JOB_NUM_NODES nodes allocate
if [ "$SLURM_JOB_NUM_NODES" -le 3 ]; then
    echo "Error: This script requires more than 3 nodes, but only $SLURM_JOB_NUM_NODES node(s) allocated"
    exit 1

fi

# SLURM_JOB_NUM_NODES = 4
#  -- RANK 0: Router + Benchmark
#  -- RANK 1: Prefill
#  -- RANK 2: Decode
#  -- RANK 3: Decode

# SLURM_JOB_NUM_NODES = 5

#  -- RANK 4: Prefill

# Print the node layout
echo "Node layout:"
echo "SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES"
echo "  -- RANK 0: Router + Benchmark"
echo "  -- RANK 1: Prefill - DEP2"
echo "  -- RANK 2: Decode - DEP8 - 1/2"
echo "  -- RANK 3: Decode - DEP8 - 2/2"
if [ "$SLURM_JOB_NUM_NODES" -eq 7 ]; then
    echo "  -- RANK 4: Prefill - DEP2"
    echo "  -- RANK 5: Prefill - DEP2"
    echo "  -- RANK 6: Prefill - DEP2"
fi


# Get all node hostnames and extract their numbers (improved parsing)
ALL_NODE_NUMS=()

# export nsys=true

while IFS= read -r hostname; do
  [ -n "$hostname" ] && ALL_NODE_NUMS+=("$hostname")
done < <(scontrol show hostnames "$SLURM_NODELIST" 2>/dev/null)

PREFILL_1_HOSTNAME="${ALL_NODE_NUMS[1]}"
DECODE_1_HOSTNAME="${ALL_NODE_NUMS[2]}"
DECODE_2_HOSTNAME="${ALL_NODE_NUMS[3]}"
PREFILL_2_HOSTNAME="${ALL_NODE_NUMS[4]}"
PREFILL_3_HOSTNAME="${ALL_NODE_NUMS[5]}"
PREFILL_4_HOSTNAME="${ALL_NODE_NUMS[6]}"

ALL_NODE_NUMS_STR="${ALL_NODE_NUMS[*]}"
DPS="4"
srun --segment $SLURM_JOB_NUM_NODES --ntasks-per-node=1 sudo nvidia-smi -ac 3996,2062

srun --segment $SLURM_JOB_NUM_NODES \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$VLLM_DIR:/scratch,$MODEL_DIR:/ds-models" \
  --container-workdir=/scratch/runs/DS-R1/fp4 \
  --mpi=pmix \
  bash -c "
RANK=0
for node in $ALL_NODE_NUMS_STR; do
    if [ \"\$node\" = \"\$(hostname -s)\" ]; then
        break
    fi
    RANK=\$((RANK + 1))
done
if [ \"\$RANK\" = 1 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_1_HOSTNAME: $PREFILL_1_HOSTNAME RANK: \$RANK DPS: 2\"
    just prefill-master-pd $PREFILL_1_HOSTNAME 2
elif [ \"\$RANK\" = 2 ]; then
    echo \"HOSTNAME: \$(hostname -s)  DECODE_1_HOSTNAME: $DECODE_1_HOSTNAME RANK: \$RANK DPS: 8\"
    just decode-master-pd $DECODE_1_HOSTNAME 8
elif [ \"\$RANK\" = 3 ]; then
    echo \"HOSTNAME: \$(hostname -s)  DECODE_2_HOSTNAME: $DECODE_2_HOSTNAME RANK: \$RANK DPS: 8\"
    just decode-worker-pd $DECODE_1_HOSTNAME 4 8
elif [ \"\$RANK\" = 4 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_2_HOSTNAME: $PREFILL_2_HOSTNAME RANK: \$RANK DPS: 2\"
    just prefill-master-pd $PREFILL_2_HOSTNAME 2
elif [ \"\$RANK\" = 5 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_3_HOSTNAME: $PREFILL_3_HOSTNAME RANK: \$RANK DPS: 2\"
    just prefill-master-pd $PREFILL_3_HOSTNAME 2
elif [ \"\$RANK\" = 6 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_4_HOSTNAME: $PREFILL_4_HOSTNAME RANK: \$RANK DPS: 2\"
    just prefill-master-pd $PREFILL_4_HOSTNAME 2
elif [ \"\$RANK\" = 0 ]; then
    while ! curl -s http://$PREFILL_1_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$PREFILL_2_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$PREFILL_3_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$PREFILL_4_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$DECODE_1_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$DECODE_2_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    echo \"vLLM is ready, creating router server...\"
    just router http://$PREFILL_1_HOSTNAME:8087 http://$PREFILL_2_HOSTNAME:8087 http://$PREFILL_3_HOSTNAME:8087 http://$PREFILL_4_HOSTNAME:8087 http://$DECODE_1_HOSTNAME:8087 http://$DECODE_2_HOSTNAME:8087 &
    echo \"router is ready, running accuracy...\"
    just accuracy 0.0.0.0  8192
    echo \"router server is ready, running benchmark...\"
    just bench 0.0.0.0 8192 20480 2048 1024 0 16
else
    echo \"Something went wrong...\"
fi
"

