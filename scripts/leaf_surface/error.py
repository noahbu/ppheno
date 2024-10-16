import numpy as np

# Ground truth and calculated values
ground_truth = np.array([703, 369, 705, 388, 421, 446, 398])
calculated = np.array([703, 320, 645, 364, 389, 425, 378])

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(ground_truth - calculated))

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((ground_truth - calculated)**2))

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((ground_truth - calculated) / ground_truth)) * 100

# R-squared (R²)
ss_res = np.sum((ground_truth - calculated) ** 2)
ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

mae, rmse, mape, r_squared

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared (R²): {r_squared:.2f}")