#!/bin/bash

# first with: chmod +x export_pointclouds.sh

# Base directory containing the training subdirectories
base_dir="/home/ubuntu/outputs/MuskMelon_C/2024-08-02"
export_base_dir="/home/ubuntu/ppheno/data/MuskMelon_C/2024-08-02"

# Log file for the export process
log_file="${export_base_dir}/PointCloud_Export_Log_$(date +'%Y-%m-%d_%H-%M-%S').txt"
echo "Starting point cloud export process" > "$log_file"
echo "Base directory: $base_dir" >> "$log_file"
echo "Export base directory: $export_base_dir" >> "$log_file"

# Loop through each subdirectory in the base directory
for model_dir in "$base_dir"/*; do
    if [ -d "$model_dir" ]; then
        # Extract the name of the subdirectory (e.g., A-1)
        dir_name=$(basename "$model_dir")
        
        # Define the output directory for the current export
        export_dir="${export_base_dir}/${dir_name}"

        # Ensure the export directory exists
        mkdir -p "$export_dir"

        # Find the latest config.yml file in the nerfacto directory
        latest_config_dir=$(find "$model_dir/nerfacto/$dir_name/nerfacto" -type d -name "2024-*" | sort | tail -n 1)
        config_file="${latest_config_dir}/config.yml"

        if [ -f "$config_file" ]; then
            # Define the name for the output point cloud
            output_name="${dir_name}_PointCloud_$(basename "$latest_config_dir").ply"
            output_file="${export_dir}/${output_name}"

            echo "Exporting point cloud for $dir_name using config file: $config_file" >> "$log_file"
            echo "Output file: $output_file" >> "$log_file"

            # Run the export command
            ns-export pointcloud \
                --load-config "$config_file" \
                --output-dir "$export_dir" \
                --num-points 1000000 \
                --remove-outliers True \
                --normal-method model_output \
                --save-world-frame True
            
            echo "Export completed for $dir_name, saved to $output_file" >> "$log_file"
        else
            echo "Config file not found for $dir_name, skipping export." >> "$log_file"
        fi
    fi
done

echo "All point cloud exports completed." >> "$log_file"
