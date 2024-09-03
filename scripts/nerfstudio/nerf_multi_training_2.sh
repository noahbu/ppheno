#!/bin/bash

# Base directory containing the training subdirectories
base_dir="/home/ubuntu/data/custom/MuskMelon_C/2024-08-02"

# Get the current date and time for logging purposes
current_datetime=$(date +%Y-%m-%d_%H-%M-%S)

# Log file name for all training sessions
log_file="${base_dir}/log_2024-08-02_${current_datetime}.log"

echo "Starting all training sessions in base directory: $base_dir" | tee -a "$log_file"
echo "Logging output to: $log_file" | tee -a "$log_file"

# Output base directory
output_base_dir="/home/ubuntu/outputs/MuskMelon_C/2024-08-02"

# Loop through each subdirectory in the base directory
for data_dir in "$base_dir"/*
do
    if [ -d "$data_dir" ]; then
        # Extract the name of the subdirectory (e.g., A-1)
        dir_name=$(basename "$data_dir")
        
        # Define the output directory for the current training session
        output_dir="${output_base_dir}/${dir_name}"

        # Create the output directory if it doesn't exist
        mkdir -p "$output_dir/nerfacto"

        echo "Starting training for directory: $data_dir with wandb run name: $dir_name" | tee -a "$log_file"

        # Record the current size of the log file
        initial_log_size=$(stat -c %s "$log_file")

        # Start the training in the background
        ns-train nerfacto \
            --output-dir "$output_dir/nerfacto" \
            --method-name nerfacto \
            --experiment-name "$dir_name" \
            --project-name "MuskMelon_2024_c_08-02" \
            --vis viewer+wandb \
            --data "$data_dir" \
            --pipeline.datamanager.train-num-rays-per-batch 1024 \
            --pipeline.model.predict-normals True \
            --mixed-precision True \
            --machine.num-devices 2 \
            --max-num-iterations 25000 \
            --logging.steps-per-log 100 \
            --viewer.quit-on-train-completion True
            &>> "$log_file" &
        
        # Get the PID of the background process
        train_pid=$!

        # Monitor the log file for the "Training Finished" message
        while true; do
            sleep 30  # Check every 30 seconds
            
            # Check the new content of the log file for the "Training Finished" message
            if tail -c +$((initial_log_size + 1)) "$log_file" | grep -q "🎉 Training Finished 🎉"; then
                echo "Training successfully completed for directory: $data_dir at $(date +%Y-%m-%d_%H-%M-%S)" | tee -a "$log_file"
                kill -SIGINT $train_pid  # Send SIGINT to terminate the process
                sleep 50
                break
            fi
            
            # If the process has ended but the message wasn't found, assume an error
            if ! kill -0 $train_pid 2>/dev/null; then
                echo "Error detected during training for directory: $data_dir at $(date +%Y-%m-%d_%H-%M-%S), skipping to next." | tee -a "$log_file"
                break
            fi
        done

        # Ensure the process is terminated
        kill -9 $train_pid 2>/dev/null
    fi
done

echo "All training sessions completed at $(date +%Y-%m-%d_%H-%M-%S)." | tee -a "$log_file"
