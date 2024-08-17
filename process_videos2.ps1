# Set the common base path
$basePath = "C:\Users\BBLab\Documents\nerf-preprocessing\data\2024-08-07"  # TODO: change here for specific folder

# Start capturing all output to a log file
$logFile = "$basePath\output\$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')-log.txt"
Start-Transcript -Path $logFile

# Set the base paths for input and output directories
$baseInputDir = "$basePath\indiv"
$baseOutputDir = "$basePath\output"   

# Get all video files in the input directory
$videoFiles = Get-ChildItem -Path $baseInputDir -File

foreach ($file in $videoFiles) {
    # Construct the full input and output paths
    $inputPath = $file.FullName
    $outputPath = Join-Path -Path $baseOutputDir -ChildPath $file.BaseName

    # Create output directory if it does not exist
    if (-not (Test-Path -Path $outputPath)) {
        New-Item -Path $outputPath -ItemType Directory
    }

    # Run the ns-process-data command
    & ns-process-data video --data $inputPath --output-dir $outputPath
}

# Final message to indicate completion
Write-Host "Processing complete. Output log saved to $logFile"

# Stop capturing the output
Stop-Transcript


################################
#single execution: ns-process-data video --data "C:\Users\BBLab\Documents\nerf-preprocessing\data\2024-08-05\indiv\C-5.mov" --output-dir "C:\Users\BBLab\Documents\nerf-preprocessing\data\2024-08-05\output\C-5"

#   Upload: 
#   Login: 
#   navigate to folder
#   sftp> put -r \Users\BBLab\Documents\nerf-preprocessing\data\2024-08-04\output