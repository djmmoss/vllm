#!/bin/bash

echo $SLURM_JOB_NUM_NODES nodes allocated
if [ "$SLURM_JOB_NUM_NODES" -le 0 ]; then
    echo "Error: This script requires at least 1 node, but only $SLURM_JOB_NUM_NODES node(s) allocated"
    exit 1
fi

# Get all node hostnames and extract their numbers (improved parsing)
ALL_NODE_NUMS=()

# export nsys=true

while IFS= read -r hostname; do
    [ -n "$hostname" ] && ALL_NODE_NUMS+=("$hostname")
done < <(scontrol show hostnames "$SLURM_NODELIST" 2>/dev/null)

MASTER_HOSTNAME="${ALL_NODE_NUMS[0]}"
ALL_NODE_NUMS_STR="${ALL_NODE_NUMS[*]}"
DPS="$((SLURM_JOB_NUM_NODES * 2))"
srun --segment $SLURM_JOB_NUM_NODES --ntasks-per-node=1 sudo nvidia-smi -ac 3996,2062

srun --segment $SLURM_JOB_NUM_NODES \
     --container-image=/lustre/fsw/coreai_devtech_all/jiahanc/meta-dsr1-gb200/images/vllm-custom.sqsh \
     --container-mounts=/lustre/fsw/coreai_devtech_all/jiahanc/meta-dsr1-gb200:/scratch,/lustre/fsw/coreai_devtech_all/siyuanf/models:/ds-models \
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

echo \"HOSTNAME: \$(hostname -s)  MASTER_HOSTNAME: $MASTER_HOSTNAME RANK: \$RANK DPS: $DPS\"
if [ \"\$RANK\" = 0 ]; then
    just prefill-master $MASTER_HOSTNAME $DPS &
    while ! curl -s http://$MASTER_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    echo \"vLLM is ready, running accuracy...\"
    just accuracy 0.0.0.0 8087
    echo \"vLLM is ready, running benchmark...\"
    just bench 0.0.0.0 8087 20480 1024 1
else
    just prefill-worker $MASTER_HOSTNAME \$((\$RANK * 4)) $DPS
fi
"
