import wandb
import pandas as pd

# Log in to wandb (if needed)
# wandb.login()
wandb.login(key='4a02e57f31edae80dac0c9c9b05207bbccd6a116')


# List of project names
project_names = [
    "MuskMelon_2024_c_08-01", "MuskMelon_2024_c_08-02", "MuskMelon_2024_c_08-03",
    "MuskMelon_2024_c_08-04", "MuskMelon_2024_c_08-05", "MuskMelon_2024_c_08-06",
    "MuskMelon_2024_c_08-07", "MuskMelon_2024_c_08-08", "MuskMelon_2024_c_08-17",
    "MuskMelon_2024_c_08-18"
]

# Initialize an empty list to store the metrics
metrics_summary = []

# Wandb API initialization
api = wandb.Api()

# Iterate through each project
for project_name in project_names:
    try:
        # Get all runs for the current project
        runs = api.runs(f"noah-bucher/{project_name}")
        
        for run in runs:
            # Extract desired metrics for each run
            run_data = {
                'project_name': project_name,
                'run_name': run.name,
                'Eval Metrics Dict / psnr': run.summary.get('Eval Metrics Dict/psnr'),
                'Eval Images Metrics / lpips': run.summary.get('Eval Images Metrics/lpips'),
                'Eval Images Metrics / psnr': run.summary.get('Eval Images Metrics/psnr'),
                'Eval Images Metrics / ssim': run.summary.get('Eval Images Metrics/ssim')
            }
            metrics_summary.append(run_data)
    except Exception as e:
        print(f"Error fetching data for project {project_name}: {e}")

# Create a pandas DataFrame to store and display the metrics
df = pd.DataFrame(metrics_summary)

# Print the summary
print(df)

# Optionally, save the summary to a CSV file
df.to_csv("musk_melon_project_metrics_summary.csv", index=False)
