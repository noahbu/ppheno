import os

# Define the base directory
base_dir = "/Users/noahbucher/Downloads/2024-07-30"  # Replace with the correct path

# Walk through the directory structure
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "point_cloud.ply":
            # Extract the folder name
            folder_name = os.path.basename(root)
            
            # Construct the new filename
            new_name = f"pc_2024-07-30_{folder_name}_dense_01.ply"
            
            # Get the full path of the current and new file
            old_file = os.path.join(root, file)
            new_file = os.path.join(root, new_name)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed {old_file} to {new_file}")

print("Renaming complete.")
