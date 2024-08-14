#!/bin/bash

# first with: chmod +x export_pointclouds.sh

# Base directory containing the training subdirectories
base_dir="/home/ubuntu/outputs/MuskMelon_C/2024-07-30"
export_base_dir="/home/ubuntu/exports/MuskMelon_C/2024-07-30"

# Loop through each subdirectory in the base directory
for model_dir in "$base_dir"/*
do
    if [ -d "$model_dir" ]; then
        # Extract the name of the subdirectory (e.g., A-1)
        dir_name=$(basename "$model_dir")
        
        # Define the output directory for the current export
        export_dir="${export_base_dir}/${dir_name}"

        # Ensure the export directory exists
        mkdir -p "$export_dir"

        # Find the config.yml file in the nerfacto directory
        config_file=$(find "$model_dir/nerfacto" -name "config.yml")

        if [ -f "$config_file" ]; then
            echo "Exporting point cloud for $dir_name using config file: $config_file"

            # Run the export command
            ns-export pointcloud \
                --load-config "$config_file" \
                --output-dir "$export_dir" \
                --num-points 1000000 \
                --remove-outliers True \
                --normal-method model_output \
                --save-world-frame True
            
            echo "Export completed for $dir_name, saved to $export_dir"
        else
            echo "Config file not found for $dir_name, skipping export."
        fi
    fi
done

echo "All point cloud exports completed."
