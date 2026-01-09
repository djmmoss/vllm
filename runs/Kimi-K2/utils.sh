#!/bin/bash
# Common utility functions for vLLM benchmark scripts

# Get file creation timestamp (or modification time as fallback)
# Usage: get_file_timestamp <file>
# Returns: timestamp in format YYYYMMDD_HHMMSS
get_file_timestamp() {
    local file="$1"
    if [ -f "$file" ]; then
        # Try to get creation time (modification time as fallback)
        local timestamp=""
        if stat -c %w "$file" 2>/dev/null | grep -q "^[0-9]"; then
            # Linux stat with birth time - format: YYYYMMDD_HHMMSS
            timestamp=$(stat -c %w "$file" 2>/dev/null | cut -d' ' -f1-2 | tr -d ':-' | tr ' ' '_' | cut -d'.' -f1)
        fi
        
        if [ -z "$timestamp" ]; then
            # Fallback to modification time - format: YYYYMMDD_HHMMSS
            timestamp=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1-2 | tr -d ':-' | tr ' ' '_' | cut -d'.' -f1)
        fi
        
        echo "$timestamp"
    fi
}

# Archive a log file to archive directory with collision handling
# Usage: archive_log_file <file> <archive_dir> <base_name>
# Example: archive_log_file "$BENCHMARK_LOG" "$ARCHIVE_DIR" "benchmark"
archive_log_file() {
    local file="$1"
    local archive_dir="$2"
    local base_name="$3"
    
    if [ ! -f "$file" ]; then
        return 0
    fi
    
    local timestamp=$(get_file_timestamp "$file")
    if [ -z "$timestamp" ]; then
        # Fallback: use current date-time if timestamp can't be determined
        timestamp=$(date +%Y%m%d_%H%M%S)
    fi
    
    local extension="${file##*.}"
    local archive_name="${base_name}_${timestamp}.${extension}"
    
    # Handle collisions by appending a number
    local counter=1
    local original_name="$archive_name"
    while [ -f "$archive_dir/$archive_name" ]; do
        archive_name="${original_name%.${extension}}_${counter}.${extension}"
        counter=$((counter + 1))
    done
    
    mv "$file" "$archive_dir/$archive_name"
    echo "Archived $file to $archive_dir/$archive_name"
}

# Cleanup function for server processes
# Usage: cleanup_server <pid_file>
# Note: This function should be called via trap: trap cleanup_server EXIT INT TERM
cleanup_server() {
    local pid_file="${PID_FILE:-}"
    if [ -z "$pid_file" ]; then
        echo "Warning: PID_FILE not set, cannot cleanup server"
        return
    fi
    
    # Stop metrics polling if it's running
    if [ -n "${METRICS_POLLING_PID:-}" ]; then
        stop_metrics_polling
    fi
    
    if [ -f "$pid_file" ]; then
        local server_pid=$(cat "$pid_file")
        if ps -p "$server_pid" > /dev/null 2>&1; then
            echo "Stopping vLLM server (PID: $server_pid)..."
            kill "$server_pid" || true
            wait "$server_pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

# Check if model path exists
# Usage: check_model_path <model_path>
# Exits with error if model path doesn't exist
check_model_path() {
    local model_path="$1"
    if [ ! -d "$model_path" ]; then
        echo "Error: Model not found at $model_path"
        echo "Please run the download script first: ../models/download_models.sh"
        exit 1
    fi
}

# Wait for server to be ready
# Usage: wait_for_server <server_pid> <server_host> <server_port> <server_log> [max_wait]
# Returns: 0 if server is ready, 1 if server failed or timeout
wait_for_server() {
    local server_pid="$1"
    local server_host="$2"
    local server_port="$3"
    local server_log="$4"
    local max_wait="${5:-1200}"  # Default 20 minutes
    
    echo "Waiting for server to be ready..."
    local waited=0
    local ready=false
    
    while [ $waited -lt $max_wait ]; do
        # Check if server process is still running
        if ! ps -p "$server_pid" > /dev/null 2>&1; then
            echo "Error: Server process died. Check $server_log for details."
            tail -n 50 "$server_log"
            return 1
        fi
        
        # Check if server is responding
        if curl -s "http://${server_host}:${server_port}/health" > /dev/null 2>&1; then
            ready=true
            break
        fi
        
        sleep 2
        waited=$((waited + 2))
        
        # Show progress every 10 seconds
        if [ $((waited % 10)) -eq 0 ]; then
            echo "  Still waiting... (${waited}s)"
        fi
    done
    
    if [ "$ready" = false ]; then
        echo "Error: Server did not become ready within $max_wait seconds."
        echo "Last 50 lines of server log:"
        tail -n 50 "$server_log"
        return 1
    fi
    
    echo "Server is ready!"
    return 0
}

# Print profiling help message
# Usage: print_profiling_help [script_name]
# Parameters:
#   script_name: Name of the script (default: uses $0)
print_profiling_help() {
    local script_name="${1:-$0}"
    echo "Usage: $script_name [--profile] [--nsys] [--profiler-cycles N] [--help]"
    echo ""
    echo "Arguments:"
    echo "  --profile              Enable torch profiler (default: disabled)"
    echo "  --nsys                 Enable nsys profiling (mutually exclusive with --profile)"
    echo "  --profiler-cycles N    Number of profiling cycles to run (default: 1)"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Modes:"
    echo "  No flags:              Normal benchmark"
    echo "  --profile:              Torch profiler enabled"
    echo "  --nsys:                Nsys profiling enabled"
    echo ""
    echo "Examples:"
    echo "  $script_name                           # Run without profiling"
    echo "  $script_name --profile                 # Run with torch profiler (1 cycle)"
    echo "  $script_name --profile --profiler-cycles 3  # Run with torch profiler (3 cycles)"
    echo "  $script_name --nsys                   # Run with nsys profiling (1 cycle)"
    echo "  $script_name --nsys --profiler-cycles 5    # Run with nsys profiling (5 cycles)"
}

# Parse profiling flags from command line arguments
# Usage: parse_profiling_flags "$@"
# Note: This function should be called before parsing other arguments
# Sets: ENABLE_TORCH_PROFILE and ENABLE_NSYS_PROFILE (boolean)
# Sets: USE_CUSTOM_VLLM (boolean, default: true)
# Sets: USE_NIGHTLY_VLLM (boolean, default: false)
# Sets: PROFILER_CYCLES (number, default: 1)
# Sets: PROFILING_REMAINING_ARGS array with remaining arguments
parse_profiling_flags() {
    # Initialize defaults
    PROFILER_CYCLES="${PROFILER_CYCLES:-1}"
    
    # Check if profiling flags are set as environment variables (for SLURM jobs)
    if [ -n "$ENABLE_TORCH_PROFILE" ] || [ -n "$ENABLE_NSYS_PROFILE" ]; then
        # Environment variables take precedence (set by SLURM submission)
        if [ "$ENABLE_TORCH_PROFILE" = "1" ] || [ "$ENABLE_TORCH_PROFILE" = "true" ]; then
            ENABLE_TORCH_PROFILE=true
        else
            ENABLE_TORCH_PROFILE=false
        fi
        if [ "$ENABLE_NSYS_PROFILE" = "1" ] || [ "$ENABLE_NSYS_PROFILE" = "true" ]; then
            ENABLE_NSYS_PROFILE=true
        else
            ENABLE_NSYS_PROFILE=false
        fi
        # Check for USE_CUSTOM_VLLM and USE_NIGHTLY_VLLM environment variables
        if [ "$USE_CUSTOM_VLLM" = "1" ] || [ "$USE_CUSTOM_VLLM" = "true" ]; then
            USE_CUSTOM_VLLM=true
        else
            USE_CUSTOM_VLLM=false
        fi
        if [ "$USE_NIGHTLY_VLLM" = "1" ] || [ "$USE_NIGHTLY_VLLM" = "true" ]; then
            USE_NIGHTLY_VLLM=true
        else
            USE_NIGHTLY_VLLM=false
        fi
        # Use PROFILER_CYCLES from environment if set, otherwise keep default from initialization
        PROFILER_CYCLES="${PROFILER_CYCLES:-1}"
        PROFILING_REMAINING_ARGS=("$@")
        return 0
    fi
    
    # Parse from command line
    ENABLE_TORCH_PROFILE=false
    ENABLE_NSYS_PROFILE=false
    USE_CUSTOM_VLLM="${USE_CUSTOM_VLLM:-true}"  # Default to true (custom image)
    USE_NIGHTLY_VLLM="${USE_NIGHTLY_VLLM:-false}"  # Default to false
    PROFILING_REMAINING_ARGS=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --profile)
                ENABLE_TORCH_PROFILE=true
                shift
                ;;
            --nsys)
                ENABLE_NSYS_PROFILE=true
                shift
                ;;
            --custom-vllm)
                USE_CUSTOM_VLLM=true
                USE_NIGHTLY_VLLM=false
                shift
                ;;
            --nightly)
                USE_NIGHTLY_VLLM=true
                USE_CUSTOM_VLLM=false
                shift
                ;;
            --profiler-cycles)
                PROFILER_CYCLES="$2"
                shift 2
                ;;
            *)
                PROFILING_REMAINING_ARGS+=("$1")
                shift
                ;;
        esac
    done
    
    # Validate that both profiling modes are not enabled simultaneously
    if [ "$ENABLE_TORCH_PROFILE" = true ] && [ "$ENABLE_NSYS_PROFILE" = true ]; then
        echo "Error: --profile and --nsys cannot be used together"
        echo "Use --help for usage information"
        exit 1
    fi
    
    # Ensure PROFILER_CYCLES is set (default to 1 if not provided)
    PROFILER_CYCLES="${PROFILER_CYCLES:-1}"
}

# Set up torch profiler directory and environment variable
# Usage: setup_torch_profiler <script_dir> [rank]
# Sets: TORCH_PROFILER_DIR and exports VLLM_TORCH_PROFILER_DIR
# Parameters:
#   script_dir: Directory for profiler output
#   rank: Optional rank number for rank-specific directory (default: no rank suffix)
setup_torch_profiler() {
    local script_dir="$1"
    local rank="${2:-}"
    if [ "$ENABLE_TORCH_PROFILE" = true ]; then
        if [ -n "$rank" ]; then
            TORCH_PROFILER_DIR="$script_dir/vllm_profile_rank${rank}"
        else
            TORCH_PROFILER_DIR="$script_dir/vllm_profile"
        fi
        mkdir -p "$TORCH_PROFILER_DIR"
        export VLLM_TORCH_PROFILER_DIR="$TORCH_PROFILER_DIR"
        echo "Torch profiler output directory: $TORCH_PROFILER_DIR"
    fi
}

# Set up nsys profiler output path
# Usage: setup_nsys_profiler <script_dir> <model_path> [rank] [test_type]
# Sets: NSYS_OUTPUT_PATH
# Parameters:
#   script_dir: Directory for nsys output
#   model_path: Path to model (used for naming)
#   rank: Optional rank number for rank-specific filename (default: no rank suffix)
#   test_type: Optional test type (prefill/decode, default: decode)
setup_nsys_profiler() {
    local script_dir="$1"
    local model_path="$2"
    local rank="${3:-}"
    local test_type="${4:-decode}"
    
    if [ "$ENABLE_NSYS_PROFILE" = true ]; then
        # Install nvtx package if not already installed (required for NVTX profiling)
        echo "Checking for nvtx package..." >&2
        if ! python3 -c "import nvtx" 2>/dev/null; then
            echo "Installing nvtx package for NVTX profiling..." >&2
            pip install --quiet nvtx 2>&1 | grep -v "^$" >&2 || {
                echo "Warning: Failed to install nvtx package. NVTX profiling may not work correctly." >&2
            }
        else
            echo "nvtx package already installed" >&2
        fi
        
        local model_name=$(basename "$model_path")
        # Convert model name to short format (e.g., "DeepSeek-R1-0528-FP4" -> "ds-fp4")
        local model_short=""
        if [[ "$model_name" =~ DeepSeek.*FP4 ]]; then
            model_short="ds-fp4"
        elif [[ "$model_name" =~ DeepSeek.*FP8 ]]; then
            model_short="ds-fp8"
        else
            # Fallback: use lowercase model name with hyphens
            model_short=$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | sed 's/[^a-z0-9-]//g')
        fi
        
        # Create nsys output directory and filename
        local nsys_output_dir="$script_dir"
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local random_input_len="${RANDOM_INPUT_LEN:-2048}"
        local random_output_len="${RANDOM_OUTPUT_LEN:-1024}"
        local max_concurrency="${MAX_CONCURRENCY:-1024}"
        local num_prompts="${NUM_PROMPTS:-1}"
        local num_nodes="${NUM_NODES:-1}"
        local rank_suffix=""
        if [ -n "$rank" ]; then
            rank_suffix="_rank${rank}"
        fi
        # Create more descriptive filename: model-testtype_nodes_inputlen_outputlen_concurrency_prompts_rank_timestamp.nsys-rep
        local nsys_output_file="${model_short}-${test_type}_n${num_nodes}_il${random_input_len}_ol${random_output_len}_mc${max_concurrency}_np${num_prompts}${rank_suffix}_${timestamp}.nsys-rep"
        NSYS_OUTPUT_PATH="$nsys_output_dir/$nsys_output_file"

        export VLLM_NVTX_SCOPES_FOR_PROFILING=1
        echo "NVTX scopes for profiling: $VLLM_NVTX_SCOPES_FOR_PROFILING"
        echo "Nsys output path: $NSYS_OUTPUT_PATH"
    fi
}

# Export profiling flags for SLURM job submission
# Usage: export_profiling_flags
# Exports: ENABLE_TORCH_PROFILE, ENABLE_NSYS_PROFILE, USE_CUSTOM_VLLM, USE_NIGHTLY_VLLM, and PROFILER_CYCLES as strings
export_profiling_flags() {
    if [ "$ENABLE_TORCH_PROFILE" = true ]; then
        export ENABLE_TORCH_PROFILE="true"
    else
        export ENABLE_TORCH_PROFILE="false"
    fi
    if [ "$ENABLE_NSYS_PROFILE" = true ]; then
        export ENABLE_NSYS_PROFILE="true"
    else
        export ENABLE_NSYS_PROFILE="false"
    fi
    if [ "${USE_CUSTOM_VLLM:-true}" = true ]; then
        export USE_CUSTOM_VLLM="true"
    else
        export USE_CUSTOM_VLLM="false"
    fi
    if [ "${USE_NIGHTLY_VLLM:-false}" = true ]; then
        export USE_NIGHTLY_VLLM="true"
    else
        export USE_NIGHTLY_VLLM="false"
    fi
    # Export PROFILER_CYCLES if set, otherwise export default of 1
    export PROFILER_CYCLES="${PROFILER_CYCLES:-1}"
}

# Get nsys container image path if nsys profiling is enabled
# Usage: get_nsys_image <project_root>
# Returns: path to nsys image or empty string
# Note: Uses custom-nsys by default, or nightly-nsys if --nightly is specified
get_nsys_image() {
    local project_root="$1"
    if [ "$ENABLE_NSYS_PROFILE" = true ]; then
        local nsys_image
        # Use custom-nsys by default, or nightly-nsys if --nightly is specified
        if [ "${USE_NIGHTLY_VLLM:-false}" = true ]; then
            nsys_image="$project_root/images/vllm-nightly-nsys.sqsh"
        else
            nsys_image="$project_root/images/vllm-custom-nsys.sqsh"
        fi
        if [ ! -f "$nsys_image" ]; then
            echo "Error: nsys-enabled container image not found: $nsys_image" >&2
            echo "Please ensure $(basename "$nsys_image") exists in images/ directory" >&2
            exit 1
        fi
        echo "$nsys_image"
    fi
}

# Get custom vLLM container image path
# Usage: get_custom_vllm_image <project_root>
# Returns: path to custom vLLM image
# Note: If --nsys is enabled, returns vllm-custom-nsys.sqsh
get_custom_vllm_image() {
    local project_root="$1"
    local custom_image
    # If nsys is enabled, use custom-nsys image
    if [ "${ENABLE_NSYS_PROFILE:-false}" = true ]; then
        custom_image="$project_root/images/vllm-custom-nsys.sqsh"
    else
        custom_image="$project_root/images/vllm-custom.sqsh"
    fi
    if [ ! -f "$custom_image" ]; then
        echo "Error: custom vLLM container image not found: $custom_image" >&2
        echo "Please ensure $(basename "$custom_image") exists in images/ directory" >&2
        exit 1
    fi
    echo "$custom_image"
}

# Get nightly vLLM container image path
# Usage: get_nightly_image <project_root>
# Returns: path to nightly vLLM image
# Note: If --nsys is enabled, returns vllm-nightly-nsys.sqsh
get_nightly_image() {
    local project_root="$1"
    local nightly_image
    # If nsys is enabled, use nightly-nsys image
    if [ "${ENABLE_NSYS_PROFILE:-false}" = true ]; then
        nightly_image="$project_root/images/vllm-nightly-nsys.sqsh"
    else
        # Try to find nightly image (check common names)
        if [ -f "$project_root/images/vllm-nightly.sqsh" ]; then
            nightly_image="$project_root/images/vllm-nightly.sqsh"
        elif [ -f "$project_root/images/vllm-nightly-aarch64.sqsh" ]; then
            nightly_image="$project_root/images/vllm-nightly-aarch64.sqsh"
        else
            # Try to find any vllm-nightly-*.sqsh file
            nightly_image=$(ls -t "$project_root/images"/vllm-nightly-*.sqsh 2>/dev/null | head -n1)
        fi
    fi
    if [ -z "$nightly_image" ] || [ ! -f "$nightly_image" ]; then
        echo "Error: nightly vLLM container image not found" >&2
        echo "Please ensure a vllm-nightly*.sqsh file exists in images/ directory" >&2
        exit 1
    fi
    echo "$nightly_image"
}

# Start vLLM server with appropriate profiling wrapper
# Usage: start_vllm_server_with_profiling <server_log> <vllm_args...>
# Returns: server PID
# Note: Relies on ENABLE_NSYS_PROFILE and VLLM_TORCH_PROFILER_DIR environment variables
start_vllm_server_with_profiling() {
    local server_log="$1"
    shift  # Remove server_log from args
    local vllm_args=("$@")
    
    # Check if nsys profiling is enabled
    if [ "$ENABLE_NSYS_PROFILE" = true ]; then
        echo "Starting vLLM server with nsys profiling wrapper" >&2
        echo "Nsys output: $NSYS_OUTPUT_PATH" >&2
        # Use PROFILER_CYCLES if set, otherwise default to 1
        local nsys_cycles="${PROFILER_CYCLES:-1}"
        echo "Nsys profiling cycles: $nsys_cycles" >&2
        # Wrap vLLM server with nsys profiler
        VLLM_TORCH_CUDA_PROFILE=1 \
        nsys profile \
            --trace=nvtx,cuda \
            --gpu-metrics-device=all \
            --trace-fork-before-exec=true \
            --cuda-graph-trace=node \
            --capture-range=cudaProfilerApi \
            --capture-range-end "repeat-shutdown:${nsys_cycles}" \
            -o "$NSYS_OUTPUT_PATH" \
            vllm serve \
            "${vllm_args[@]}" \
            > "$server_log" 2>&1 &
    elif [ -n "$VLLM_TORCH_PROFILER_DIR" ]; then
        echo "Starting vLLM server with torch profiler enabled" >&2
        echo "Torch profiler directory: $VLLM_TORCH_PROFILER_DIR" >&2
        # For torch profiling, VLLM_TORCH_PROFILER_DIR should already be set by setup_torch_profiler
        # The profiler will be initialized automatically by vLLM when this env var is set
        vllm serve \
            "${vllm_args[@]}" \
            > "$server_log" 2>&1 &
    else
        echo "Starting vLLM server (no profiling)" >&2
        vllm serve \
            "${vllm_args[@]}" \
            > "$server_log" 2>&1 &
    fi
    
    echo $!
}

# Start profiler control process (runs profiling cycles indefinitely)
# Usage: start_profiler_control <server_host> <server_port> [initial_delay] [profiler_duration] [cycle_delay]
# Sets: PROFILE_CONTROL_PID (global variable)
# Parameters:
#   server_host: Server hostname (use "localhost" not "0.0.0.0")
#   server_port: Server port
#   initial_delay: Initial delay in seconds before first cycle (default: 180)
#   profiler_duration: How long to keep profiler running per cycle in seconds (default: 30)
#   cycle_delay: Delay between cycles in seconds (default: 60)
start_profiler_control() {
    local server_host="$1"
    local server_port="$2"
    local initial_delay="${3:-240}"  # Default 4 minutes
    local profiler_duration="${4:-10}"  # Default 30 seconds
    
    local server_url="http://${server_host}:${server_port}"
    
    (
        echo "Profiler control process started (PID: $$)" >&2
        echo "Waiting ${initial_delay} seconds before starting first profiling cycle..." >&2
        sleep "$initial_delay"
        
        echo "Starting profiler (cycle $cycle)..." >&2
        if curl -X POST -f -s "$server_url/start_profile" > /dev/null 2>&1; then
            echo "Profiler started successfully (cycle $cycle)" >&2
        else
            echo "Warning: Failed to start profiler (cycle $cycle)" >&2
        fi
        sleep "$profiler_duration"
        echo "Stopping profiler (cycle $cycle)..." >&2
        if curl -X POST -f -s "$server_url/stop_profile" > /dev/null 2>&1; then
            echo "Profiler stopped successfully (cycle $cycle)" >&2
        else
            echo "Warning: Failed to stop profiler (cycle $cycle)" >&2
        fi
    ) &
    PROFILE_CONTROL_PID=$!
    echo "Profiler control process PID: $PROFILE_CONTROL_PID" >&2
}

# Wait for profiler control process to finish
# Usage: wait_for_profiler_control [pid]
# Parameters:
#   pid: PID of profiler control process (default: uses PROFILE_CONTROL_PID if set)
wait_for_profiler_control() {
    local pid="${1:-${PROFILE_CONTROL_PID:-}}"
    if [ -z "$pid" ]; then
        echo "Warning: No profiler control PID provided" >&2
        return 0
    fi
    
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Waiting for profiler control process to finish..." >&2
        wait "$pid" 2>/dev/null || true
        echo "Profiler control process completed" >&2
    fi
}

# Start metrics polling process
# Usage: start_metrics_polling <server_host> <server_port> <output_file>
# Sets: METRICS_POLLING_PID (global variable)
# Parameters:
#   server_host: Server hostname (use "localhost" not "0.0.0.0")
#   server_port: Server port
#   output_file: File to append metrics to
start_metrics_polling() {
    local server_host="$1"
    local server_port="$2"
    local output_file="$3"
    
    local server_url="http://${server_host}:${server_port}"
    
    (
        echo "Metrics polling process started (PID: $$)" >&2
        echo "Polling $server_url/metrics every 30 seconds..." >&2
        echo "Output file: $output_file" >&2
        
        # Create output file with header if it doesn't exist
        if [ ! -f "$output_file" ]; then
            echo "# Metrics polling started at $(date -Iseconds)" > "$output_file"
            echo "# Polling interval: 30 seconds" >> "$output_file"
            echo "" >> "$output_file"
        fi
        
        # Poll metrics every 30 seconds
        while true; do
            local timestamp=$(date -Iseconds)
            echo "" >> "$output_file"
            echo "# Timestamp: $timestamp" >> "$output_file"
            echo "---" >> "$output_file"
            
            if ! curl -s -f "$server_url/metrics" >> "$output_file" 2>&1; then
                echo "Warning: Failed to poll metrics at $timestamp" >&2
                echo "# Error: Failed to fetch metrics" >> "$output_file"
            fi
            
            sleep 30
        done
    ) &
    METRICS_POLLING_PID=$!
    echo "Metrics polling process PID: $METRICS_POLLING_PID" >&2
}

# Stop metrics polling process
# Usage: stop_metrics_polling [pid]
# Parameters:
#   pid: PID of metrics polling process (default: uses METRICS_POLLING_PID if set)
stop_metrics_polling() {
    local pid="${1:-${METRICS_POLLING_PID:-}}"
    if [ -z "$pid" ]; then
        echo "Warning: No metrics polling PID provided" >&2
        return 0
    fi
    
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping metrics polling process (PID: $pid)..." >&2
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        echo "Metrics polling process stopped" >&2
    fi
}

# Run profiling benchmark (warmup + single prompt with profiling)
# Usage: run_profiling_benchmark <model_path> <server_host> <server_port> <benchmark_log> <random_input_len> <random_output_len> <random_range_ratio> <max_concurrency>
run_profiling_benchmark() {
    local model_path="$1"
    local server_host="$2"
    local server_port="$3"
    local benchmark_log="$4"
    local random_input_len="$5"
    local random_output_len="$6"
    local random_range_ratio="$7"
    local max_concurrency="$8"
    local num_prompts=1
    
    # First run: benchmark without profiling (warmup)
    echo "Starting benchmark (without profiling - warmup)..."
    echo "Benchmark log: $benchmark_log"
    echo ""
    
    vllm bench serve \
        --backend vllm \
        --model "$model_path" \
        --host "$server_host" \
        --port "$server_port" \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --random-input-len "$random_input_len" \
        --random-output-len "$random_output_len" \
        --random-range-ratio "$random_range_ratio" \
        2>&1 | tee -a "$benchmark_log"
    
    echo ""
    echo "Warmup benchmark completed!"
    echo ""
    
    # Second run: benchmark with profiling (single prompt)
    echo "Starting benchmark with profiling (single prompt)..."
    echo ""
    
    vllm bench serve \
        --backend vllm \
        --model "$model_path" \
        --host "$server_host" \
        --port "$server_port" \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --random-input-len "$random_input_len" \
        --random-output-len "$random_output_len" \
        --random-range-ratio "$random_range_ratio" \
        --profile \
        2>&1 | tee -a "$benchmark_log"
}
