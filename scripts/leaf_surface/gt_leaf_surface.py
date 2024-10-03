import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Load the image
image_path = '/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/leaf_area/gt/musk_leaf/Scan.png'
image = cv2.imread(image_path)

# Step 1: Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 2: Define the green color range in HSV to include brown spots
lower_bound = np.array([25, 30, 30])  # Lower bound to capture leaf parts
upper_bound = np.array([85, 255, 255])  # Upper bound to capture leaf parts

# Step 3: Create a mask to filter out the background
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Step 4: Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Visualize the contours before clustering for debugging
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

# Convert BGR to RGB for display with Matplotlib
contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

# Plot and visualize the contours
plt.imshow(contour_image_rgb)
plt.title('Detected Contours Before Clustering')
plt.axis('on')  # Ensure axes are shown for debugging
plt.show()

# Step 5: Extract contour centers (centroids) for clustering
contour_centroids = []
valid_contours = []

for idx, contour in enumerate(contours):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour_centroids.append([cX, cY])
        valid_contours.append(contour)  # Only add valid contours

contour_centroids = np.array(contour_centroids)

# Ensure we have enough contours before clustering
num_clusters = 4  # 5 leaf parts + 1 coin
if len(contour_centroids) < num_clusters:
    print(f"Only {len(contour_centroids)} valid contours were detected.")
else:
    # Step 6: Cluster the centroids into 6 groups (5 leaves + 1 coin)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(contour_centroids)
    labels = kmeans.labels_

    # Step 7: Create a new mask for the clustered leaves and the coin
    clustered_mask = np.zeros_like(mask)

    # Safeguard the indexing to ensure that labels map correctly to contours
    for i in range(num_clusters):
        cluster_contours = [valid_contours[j] for j in range(len(valid_contours)) if labels[j] == i]
        cv2.drawContours(clustered_mask, cluster_contours, -1, (255), thickness=cv2.FILLED)

    # Step 8: Segment the plant using the clustered mask
    segmented_plant = cv2.bitwise_and(image, image, mask=clustered_mask)

    # Step 9: Visualize the clustered mask and segmented plant
    plt.subplot(1, 2, 1)
    plt.imshow(clustered_mask, cmap='gray')
    plt.title('Clustered Mask')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(segmented_plant, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Plant')

    # Save the intermediate mask plot in the same directory with '_mask' suffix
    directory, filename = os.path.split(image_path)
    file_name_wo_ext, ext = os.path.splitext(filename)
    mask_filename = f"{file_name_wo_ext}_mask{ext}"
    mask_save_path = os.path.join(directory, mask_filename)
    
    plt.savefig(mask_save_path)
    print(f"Saved mask plot to {mask_save_path}")
    
    plt.show()

    # Step 10: Calculate the surface area in pixels
    plant_area_pixels = np.sum(clustered_mask == 255)

    # Convert the pixel area to real-world units
    resolution_dpi = 300  # Adjust based on your scanner resolution
    plant_area_in_square_inches = plant_area_pixels / (resolution_dpi ** 2)
    plant_area_in_square_cm = plant_area_in_square_inches * (2.54 ** 2)

    # Print the calculated surface area in square cm
    print(f"Plant surface area in square cm: {plant_area_in_square_cm:.2f} cm²")

    # # Step 11: Save visualization of original scan with scales and title
    # fig, ax = plt.subplots()
    # ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # ax.set_title('Original Scan with Scales')
    # ax.set_xlabel('X-axis (pixels)')
    # ax.set_ylabel('Y-axis (pixels)')

    # # Add a grid for better scale visualization (optional)
    # ax.grid(True)

    # # Save this plot in the same directory with '_original' suffix
    # original_filename = f"{file_name_wo_ext}_original{ext}"
    # original_save_path = os.path.join(directory, original_filename)
    
    # fig.savefig(original_save_path, dpi=300)  # Save the figure with 300 DPI for clarity
    # print(f"Saved original scan visualization to {original_save_path}")

    # plt.show()

    # Step 11: Save visualization of original scan alongside the segmented plant
    fig, axs = plt.subplots(1, 2, figsize=(6, 5))  # Create two subplots side by side

    # Plot the original scan on the left
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Scan')
    axs[0].set_xlabel('X-axis (pixels)')
    axs[0].set_ylabel('Y-axis (pixels)')
    axs[0].grid(True)

    # Plot the segmented plant on the right
    axs[1].imshow(clustered_mask, cmap='gray')
    axs[1].set_title('Masked Scan')
    axs[1].set_xlabel('X-axis (pixels)')
    axs[1].grid(True)

    # Remove Y-axis tick labels from the second plot but keep the grid lines and ticks
    axs[1].set_yticklabels([])  # Remove Y-axis labels but keep the tick marks

    # Save the combined plot in the same directory with '_combined' suffix
    combined_filename = f"{file_name_wo_ext}_combined{ext}"
    combined_save_path = os.path.join(directory, combined_filename)
    fig.savefig(combined_save_path, dpi=300)  # Save the figure with 300 DPI for clarity
    print(f"Saved combined scan visualization to {combined_save_path}")

    plt.show()
