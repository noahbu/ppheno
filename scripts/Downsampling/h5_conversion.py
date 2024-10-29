import numpy as np
import h5py

# def convert_txt_to_h5(txt_file, h5_file):
#     # Load the .txt file
#     data = np.loadtxt(txt_file)
    
#     # Create an HDF5 file
#     with h5py.File(h5_file, 'w') as hf:
#         # Write the data into the HDF5 file, you can give the dataset a name like 'point_cloud_data'
#         hf.create_dataset('data', data=data)
    
#     print(f"Successfully converted {txt_file} to {h5_file}")

# # Example usage
# txt_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/semantic_segemntation/Downsampling/downsampled/4096_cleaned_normals_processed_s_pc_C-3_2024-08-07_dense_02.txt'  # Replace with your .txt file path
# h5_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/semantic_segemntation/Downsampling/downsampled/4096_cleaned_normals_processed_s_pc_C-3_2024-08-07_dense_02.h5'  # The .h5 output file path

# convert_txt_to_h5(txt_file, h5_file)


# Path to your .h5 file
h5_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/semantic_segemntation/Downsampling/downsampled/4096_cleaned_normals_processed_s_pc_C-3_2024-08-07_dense_02.h5'

# Open and inspect the .h5 file
with h5py.File(h5_file, 'r') as f:
    print("Keys in the HDF5 file:", list(f.keys()))
