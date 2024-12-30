# Deep Vectorization of Technical Drawings

## Overview

This project, **Deep Vectorization of Technical Drawings**, aims to process, analyze, and vectorize technical drawings using advanced deep learning techniques. The repository includes tools for data preparation, model training, evaluation, and result visualization. Various neural network models, such as U-Net and advanced transformer-based models, are implemented to predict line coordinates and confidence scores.

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Testing and Visualization](#testing-and-visualization)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Multiple Models**: Includes U-Net, regression-based models, and transformer-based models.
- **Data Augmentation**: Generates synthetic datasets and augments existing data.
- **Post-Processing**: Implements skeletonization and line detection for vectorization.
- **Visualization**: Visualize ground truth, predictions, and SVG overlays.
- **Documentation**: Automatically generated using Doxygen.

---

## Folder Structure

```plaintext
src/                      # Source code
  ├── data_loading/       # Scripts for loading and parsing data
  ├── data_preprocessing/ # Scripts for preprocessing and augmentation
  ├── models/             # Model architectures (U-Net, transformers, etc.)
  ├── training/           # Model training scripts
  ├── post_processing/    # Post-processing and vectorization tools
  ├── visualization/      # Scripts for visualizing results
  └── utils/              # Utility functions

results/                  # Results (SVGs, predictions, metrics)
data/                     # Raw and processed datasets
docs/                     # Doxygen-generated documentation
requirements.txt          # Python dependencies
README.md                 # Project overview (this file)
```

---

## Requirements

This project is implemented in Python. Ensure the following dependencies are installed:

- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- svgwrite
- scikit-image
- scikit-learn
- Doxygen (for documentation generation)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/deep-vectorization-technical-drawings.git
   cd deep-vectorization-technical-drawings
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Data:**
   Place your raw datasets in the `data/` folder. Use the `line_drawing_generation.py` script to generate synthetic datasets if needed.

4. **Generate Documentation (Optional):**
   Install Doxygen and run:
   ```bash
   doxygen Doxyfile
   ```

---

## Usage

### Data Preparation

1. **Generate Synthetic Data:**
   Use the `line_drawing_generation.py` script to create SVG, PNG, and PDF datasets:
   ```bash
   python src/line_drawing_generation.py
   ```

2. **Preprocess Data:**
   Use scripts in the `data_preprocessing/` directory to parse SVGs and prepare input-target pairs.

### Model Training

1. **Select a Model:**
   Choose from U-Net, advanced transformer-based models, or regression-based models. Edit `main.py` to specify the model you wish to train.

2. **Train the Model:**
   Run the training script:
   ```bash
   python src/main.py
   ```

### Testing and Visualization

1. **Test the Model:**
   Use the `process_test_results.py` script to process test data and generate vectorized SVG outputs:
   ```bash
   python src/process_test_results.py
   ```

2. **Visualize Results:**
   Predicted masks, skeletonized lines, and SVG overlays can be visualized using `visualization.py`.

---

## Documentation

Comprehensive project documentation is generated using Doxygen. Navigate to the `docs/` folder to view the generated HTML documentation.

---

## Contributing

Contributions are welcome! If you encounter a bug or have suggestions, feel free to open an issue or submit a pull request.

### How to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your branch.
4. Submit a pull request describing your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

