# Set the base paths for input and output directories
$baseInputDir = "C:\Users\BBLab\Documents\nerf-preprocessing\data\2024-07-30\indiv"
$baseOutputDir = "C:\Users\BBLab\Documents\nerf-preprocessing\data\2024-07-30\output"

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

    #single execution: ns-process-data video --data "C:\Users\BBLab\Documents\nerf-preprocessing\data\2024-08-03\indiv\C-4.mov" --output-dir "C:\Users\BBLab\Documents\nerf-preprocessing\data\2024-08-03\output\C-4-2"
}
