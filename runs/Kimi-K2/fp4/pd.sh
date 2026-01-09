# Fixed configuration: 2 prefill (TP=8 each, hybrid-lb) + 1 decode (TP=8, hybrid-lb)
# Total: 7 nodes (4 prefill + 2 decode + 1 router)

echo $SLURM_JOB_NUM_NODES nodes allocated
if [ "$SLURM_JOB_NUM_NODES" -lt 7 ]; then
    echo "Error: This script requires 7 nodes, but only $SLURM_JOB_NUM_NODES node(s) allocated"
    exit 1
fi

# Node layout (7 nodes):
#   Rank 0: Prefill 0 master (TP=8, GPUs 0-3)
#   Rank 1: Prefill 0 worker (TP=8, GPUs 4-7)
#   Rank 2: Prefill 1 master (TP=8, GPUs 0-3)
#   Rank 3: Prefill 1 worker (TP=8, GPUs 4-7)
#   Rank 4: Decode master (TP=8, GPUs 0-3)
#   Rank 5: Decode worker (TP=8, GPUs 4-7)
#   Rank 6: Router

echo "Node layout:"
echo "SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES"
echo "  -- RANK 0: Prefill 0 master (DP=8)"
echo "  -- RANK 1: Prefill 0 worker (DP=8)"
echo "  -- RANK 2: Prefill 1 master (DP=8)"
echo "  -- RANK 3: Prefill 1 worker (DP=8)"
echo "  -- RANK 4: Decode master (DP=8)"
echo "  -- RANK 5: Decode worker (DP=8)"
echo "  -- RANK 6: Router + Benchmark"

# Get all node hostnames
ALL_NODE_NUMS=()

while IFS= read -r hostname; do
    [ -n "$hostname" ] && ALL_NODE_NUMS+=("$hostname")
done < <(scontrol show hostnames "$SLURM_NODELIST" 2>/dev/null)

PREFILL0_MASTER="${ALL_NODE_NUMS[0]}"
PREFILL0_WORKER="${ALL_NODE_NUMS[1]}"
PREFILL1_MASTER="${ALL_NODE_NUMS[2]}"
PREFILL1_WORKER="${ALL_NODE_NUMS[3]}"
DECODE_MASTER_HOSTNAME="${ALL_NODE_NUMS[4]}"
DECODE_WORKER_HOSTNAME="${ALL_NODE_NUMS[5]}"

echo "Prefill 0: master=$PREFILL0_MASTER, worker=$PREFILL0_WORKER"
echo "Prefill 1: master=$PREFILL1_MASTER, worker=$PREFILL1_WORKER"
echo "Decode: master=$DECODE_MASTER_HOSTNAME, worker=$DECODE_WORKER_HOSTNAME"

ALL_NODE_NUMS_STR="${ALL_NODE_NUMS[*]}"
srun --segment $SLURM_JOB_NUM_NODES --ntasks-per-node=1 sudo nvidia-smi -ac 3996,2062

srun --segment $SLURM_JOB_NUM_NODES \
    --container-image=/lustre/fsw/coreai_devtech_all/jiahanc/meta-dsr1-gb200/images/vllm-custom.sqsh \
    --container-mounts=/lustre/fsw/coreai_devtech_all/jiahanc/meta-dsr1-gb200:/scratch,/lustre:/lustre \
    --container-workdir=/scratch/runs/Kimi-K2/fp4 \
    --mpi=pmix \
    bash -c "
# Export hostname variables
export PREFILL0_MASTER=$PREFILL0_MASTER
export PREFILL0_WORKER=$PREFILL0_WORKER
export PREFILL1_MASTER=$PREFILL1_MASTER
export PREFILL1_WORKER=$PREFILL1_WORKER
export DECODE_MASTER_HOSTNAME=$DECODE_MASTER_HOSTNAME
export DECODE_WORKER_HOSTNAME=$DECODE_WORKER_HOSTNAME

# Calculate current rank
RANK=0
for node in $ALL_NODE_NUMS_STR; do
    if [ \"\$node\" = \"\$(hostname -s)\" ]; then
        break
    fi
    RANK=\$((RANK + 1))
done

# Fixed node allocation for 2 prefill (TP=8 each, 2 nodes each) + 1 decode (TP=8, 2 nodes)
TOTAL_PREFILL_NODES=4
DECODE_START_RANK=4
ROUTER_START_RANK=6

# Determine node role based on rank
if [ \"\$RANK\" -lt \"\$TOTAL_PREFILL_NODES\" ]; then
    NODE_TYPE=\"prefill\"
    INSTANCE_ID=\$((RANK / 2))
    LOCAL_RANK=\$((RANK % 2))
elif [ \"\$RANK\" -lt \"\$ROUTER_START_RANK\" ]; then
    NODE_TYPE=\"decode\"
    INSTANCE_ID=0
    LOCAL_RANK=\$((RANK - DECODE_START_RANK))
else
    NODE_TYPE=\"router\"
    INSTANCE_ID=0
    LOCAL_RANK=0
fi

IS_MASTER=\$([[ \$LOCAL_RANK -eq 0 ]] && echo 1 || echo 0)

echo \"HOSTNAME: \$(hostname -s) RANK: \$RANK NODE_TYPE: \$NODE_TYPE INSTANCE_ID: \$INSTANCE_ID LOCAL_RANK: \$LOCAL_RANK IS_MASTER: \$IS_MASTER\"

# Start the appropriate service based on node type
if [ \"\$NODE_TYPE\" = \"prefill\" ]; then
    if [ \$IS_MASTER -eq 1 ]; then
        case \$INSTANCE_ID in
            0)
                echo \"Starting prefill 0 master with TP=8 (2 nodes)\"
                just prefill-master-pd $PREFILL0_MASTER 8
                ;;
            1)
                echo \"Starting prefill 1 master with TP=8 (2 nodes)\"
                just prefill-master-pd-node1 $PREFILL1_MASTER 8
                ;;
        esac
    else
        DPSR=4
        case \$INSTANCE_ID in
            0)
                echo \"Starting prefill 0 worker (LOCAL_RANK=\$LOCAL_RANK, DPSR=\$DPSR) with TP=8\"
                just prefill-worker-pd $PREFILL0_MASTER \$DPSR 8
                ;;
            1)
                echo \"Starting prefill 1 worker (LOCAL_RANK=\$LOCAL_RANK, DPSR=\$DPSR) with TP=8\"
                just prefill-worker-pd $PREFILL1_MASTER \$DPSR 8
                ;;
        esac
    fi
elif [ \"\$NODE_TYPE\" = \"decode\" ]; then
    if [ \$IS_MASTER -eq 1 ]; then
        echo \"Starting decode master with TP=8 (2 nodes)\"
        just decode-master-pd $DECODE_MASTER_HOSTNAME 8
    else
        DPSR=4
        echo \"Starting decode worker (LOCAL_RANK=\$LOCAL_RANK, DPSR=\$DPSR) with TP=8\"
        just decode-worker-pd $DECODE_MASTER_HOSTNAME \$DPSR 8
    fi
elif [ \"\$NODE_TYPE\" = \"router\" ]; then
    echo \"Router node: Setting up router and benchmark\"
    
    # Wait for all prefill instances to be ready
    for instance in 0 1; do
        PREFILL_MASTER_VAR=\"PREFILL\${instance}_MASTER\"
        PREFILL_WORKER_VAR=\"PREFILL\${instance}_WORKER\"
        PREFILL_MASTER=\"\${!PREFILL_MASTER_VAR}\"
        PREFILL_WORKER=\"\${!PREFILL_WORKER_VAR}\"
        
        while ! curl -s http://\$PREFILL_MASTER:8087/health > /dev/null 2>&1; do
            sleep 10
        done
        echo \"Prefill \$instance master (\$PREFILL_MASTER) is ready\"
        
        while ! curl -s http://\$PREFILL_WORKER:8087/health > /dev/null 2>&1; do
            sleep 10
        done
        echo \"Prefill \$instance worker (\$PREFILL_WORKER) is ready\"
    done
    
    # Wait for decode master to be ready
    while ! curl -s http://$DECODE_MASTER_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 10
    done
    echo \"Decode master ($DECODE_MASTER_HOSTNAME) is ready\"
    
    # Wait for decode worker to be ready
    while ! curl -s http://$DECODE_WORKER_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 10
    done
    echo \"Decode worker ($DECODE_WORKER_HOSTNAME) is ready\"
    
    echo \"Starting router server with hybrid-lb (4 prefill + 2 decode endpoints)...\"
    just router \\
        http://$PREFILL0_MASTER:8087 \\
        http://$PREFILL0_WORKER:8087 \\
        http://$PREFILL1_MASTER:8087 \\
        http://$PREFILL1_WORKER:8087 \\
        http://$DECODE_MASTER_HOSTNAME:8087 \\
        http://$DECODE_WORKER_HOSTNAME:8087 &
    
    while ! curl -s http://0.0.0.0:8123/health > /dev/null 2>&1; do
        sleep 10
    done
    
    source ../utils.sh
    
    echo \"Running benchmark...\"
    just bench 0.0.0.0 8123 4096 2048 1024
fi
"
