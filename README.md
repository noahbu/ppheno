# Plant Phenotyping for Musk Melon Plants

This project focuses on the phenotyping of musk melon plants using dense point clouds generated with Neural Radiance Fields (NeRFs), followed by post-processing to extract meaningful phenotypic data.

## 🌱 Project Overview

Accurate phenotyping is crucial for understanding plant characteristics and improving crop yields. This project leverages NeRFs to create detailed 3D representations of musk melon plants, enabling comprehensive analysis of their physical traits such as volume, surface area, and other morphological features.

## 🗂️ Repository Structure

- `.vscode/`: Configuration files for Visual Studio Code to ensure a consistent development environment.
- `clean_pointcloud/`: Contains scripts and tools for processing and cleaning the raw point cloud data obtained from NeRFs.
- `scripts/`: Includes various utility scripts to facilitate data preprocessing, analysis, and conversions.
- `src/`: Core source code for the project, including:
  - NeRF implementation for generating point clouds.
  - Analysis modules for extracting plant phenotypes.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `README.md`: Project overview and setup guide (this file).

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip (Python package manager)
- Git

## 🛠️ Installation

This project uses Python and Jupyter notebooks. The primary dependencies can be managed using `pip`.

### 1. Clone the Repository

```bash
git clone https://github.com/noahbu/ppheno.git
cd ppheno
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

There is no `requirements.txt` provided, but based on the Jupyter notebooks and scripts, you will likely need:

```bash
pip install numpy pandas matplotlib open3d jupyter
```

If you plan to work with NeRFs or the cleaning pipeline, you may also need:

```bash
pip install scikit-learn opencv-python
```

### 4. Run Jupyter Notebook

Most of the analysis is driven from notebooks. To start:

```bash
jupyter notebook
```

Then open the relevant `.ipynb` files to run the point cloud analysis and visualization.

---

🧠 Tip: If you're missing packages when running a notebook, Jupyter will usually tell you in the error log. You can install them on the fly with `!pip install <package-name>` in a cell.


## ⚙️ Usage

### 1. Generate Point Clouds using NeRF

- Use NeRF scripts in `src/` to generate dense 3D point clouds of melon plants from image data.

### 2. Clean the Point Clouds

- Navigate to `clean_pointcloud/`:
  ```bash
  python clean.py --input <path_to_raw_pointcloud> --output <path_to_cleaned_output>
  ```

### 3. Run Phenotypic Analysis

- Use the tools in `src/` to extract relevant phenotypic information:
  ```bash
  python analyze.py --input <cleaned_pointcloud>
  ```

### 4. Utility Scripts

- Use `scripts/` for various helper tasks such as format conversion, batch processing, etc.

## 📁 Data

The dataset used for this project includes multi-angle images of musk melon plants captured in controlled environments. Point clouds are reconstructed using NeRFs.

The NeRF reconstruction requires preprocessing of the video which was done with COLMAP, which needs to be installed for processing.

*Note: Data files are not included in the repository due to size. Contact me for access.*

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

- Fork this repo and submit a pull request
- Report bugs or request features via GitHub Issues
- Improve documentation and examples


## 🙌 Acknowledgments

- [NeRF](https://github.com/bmild/nerf) for the foundational 3D reconstruction model.
- Research in plant phenotyping and computer vision communities.

---

*Built for researchers, by researchers — enabling better plant trait analysis through cutting-edge vision models.*
