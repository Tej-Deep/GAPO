#!/bin/bash

# Configuration
export PYTHONNOUSERSITE=1
export HF_TOKEN="<Add token here>"

MODEL_BASE_PATH="<Path to checkpoint folder>"
dataset_list=("amc23" "aime24" "math" "college_math" "minerva_math" "olympiadbench")
gpu_list=(5 6 7)
NUM_GPUS=${#gpu_list[@]}
# models=("meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "Qwen/Qwen3-1.7B" "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "microsoft/Phi-4-mini-instruct")
models=("Phi-4-mini")
# models=()
types=("sft" "dpo" "sdpo" "grpo-sdpo")
prompts=("base" "boxed")

# =================================================
# 🔧 SETUP GPU TOKEN QUEUE (SEMAPHORE)
# =================================================
# Create a temporary named pipe (FIFO) to manage GPU tokens
FIFO_PATH="/tmp/gpu_fifo_$$"
mkfifo "$FIFO_PATH"

# Attach File Descriptor 3 to this FIFO for reading and writing
exec 3<>"$FIFO_PATH"

# Clean up the file path immediately (file descriptor stays open)
rm "$FIFO_PATH"

# ⚡️ FILL THE BUCKET: Push all available GPU IDs into the pipe
for gpu in "${gpu_list[@]}"; do
    echo "$gpu" >&3
done

# =================================================
# 🚀 PHASE 1: EVALUATION LOOP
# =================================================
echo "==========================================="
echo "🚀 PHASE 1: Launching evaluations using Token Bucket Strategy"
echo "==========================================="

for model in "${models[@]}"; do
    for type in "${types[@]}"; do
        for prompt in "${prompts[@]}"; do
            expt="$model-prm800k-mini-$type/checkpoint-72"
            for dataset in "${dataset_list[@]}"; do
                
                # 1. EXISTENCE CHECK
                # We do this BEFORE grabbing a GPU token to avoid wasting queue time
                OUTPUT_FILE="./eval_results/$expt-$prompt/$dataset/result-0-None.json"
                if [ -f "$OUTPUT_FILE" ]; then
                    echo "⏩ SKIPPING: $expt | $dataset (Result exists)"
                    continue
                fi
    
                # 2. ACQUIRE GPU TOKEN
                # "read" will BLOCK here if the pipe is empty (all GPUs busy).
                # It waits until a background job finishes and writes back to FD 3.
                read -u 3 gpu_id
    
                # 3. LAUNCH BACKGROUND JOB
                # We run a subshell (...) in the background &
                (
                    echo "⚡️ STARTING: $expt | $dataset on GPU $gpu_id"
                    
                    # --- The Actual Work ---
                    CUDA_VISIBLE_DEVICES=$gpu_id python run_eval.py \
                        --policy_model_path $MODEL_BASE_PATH/$expt \
                        --data "$dataset" \
                        --output_dir ./eval_results/$expt-$prompt/$dataset \
                        --batch_size 64 \
                        --prompt $prompt
                    
                    # Capture exit code to warn user if python crashed
                    exit_code=$?
                    if [ $exit_code -ne 0 ]; then
                        echo "❌ FAILURE: $expt-$prompt | $dataset on GPU $gpu_id (Exit code $exit_code)"
                    fi
    
                    # --- CLEANUP & RETURN TOKEN ---
                    # Optional: Sleep to let VLLM/PyTorch clean up memory completely
                    sleep 5 
                    
                    # WRITE THE GPU ID BACK TO THE PIPE
                    # This wakes up the main loop to launch the next job
                    echo "$gpu_id" >&3
                ) &
    
            done
        done
    done
done

# =================================================
# 🏁 WAIT FOR ALL JOBS
# =================================================
echo "⏳ Main loop finished. Waiting for remaining background jobs..."
wait
echo "✅ All evaluations complete."

# Close the file descriptor
exec 3>&-

# =================================================
# 📊 PHASE 2: COMPILATION LOOP
# =================================================
echo "==========================================="
echo "📊 PHASE 2: Compiling results"
echo "==========================================="

for model in "${models[@]}"; do
    for type in "${types[@]}"; do
        for prompt in "${prompts[@]}"; do
            # expt="$model-$type"
            expt="$model-prm800k-mini-$type/checkpoint-72-$prompt"
            
            if [ -d "./eval_results/$expt" ]; then
                echo "--- Compiling results for $expt ---"
                python eval_results.py \
                    --results_dir ./eval_results/$expt
            fi
        done
    done
done

echo -e "\n--- 🎉 All operations complete. ---"