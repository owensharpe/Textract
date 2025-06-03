# Textract
An AI/ML project to convert handwritten mathematical expressions into LaTeX code using the CROHME 2023 dataset.

## Project Overview

This project implements a three-stage approach for handwritten math recognition:
1. **File Type Conversion**: Alteration of .inkml file architecture from CROHME data into .png formats for easier reading.
2. **Symbol Detection**: CNN-based detection and segmentation of individual mathematical symbols.
3. **Structure Recognition**: Reconstruction of LaTeX expressions from detected symbols and their spatial relationships.

## Project Structure

### Setup
#### Prerequisites
- Python 3.8+
- Required packages:
```bash
pip install numpy matplotlib pillow tqdm
```

#### Downloading Data
Please visit the [CROHME Website](https://crohme2023.ltu-ai.dev/data-tools/) to learn more about the dataset and the [Downloader](https://zenodo.org/records/8428035) to download the required dataset. Please download the "CROHME23.zip" file (~1.8 GB). Place this zip file in the "data" folder within the repo:

For example,
```
data/
└── CROHME23.zip
```

#### Some Information about the Data
The CROHME 2023 dataset includes:
- Real handwritten expressions from previous CROHME competitions
- 150k+ artificially generated expressions
- Both online (stroke) and offline (image) data
- Train/validation/test splits
Note: Some artificial data files may fail to parse due to encoding issues. This is normal and doesn't affect the core dataset quality.

### Phase 1: Converting .inkml Files into .png Files
Please run this command to convert the data file types.
```bash
python data_extraction/convert_to_image.py
```
