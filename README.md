# Oil Spill Detection Project

This repository contains the implementation of an oil spill detection system using the DeepLabV3+ model with a ResNet101 backbone. The project includes a Jupyter Notebook for exploration, a dataset of labeled images, and a pre-trained model for making predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Running the Jupyter Notebook](#running-the-jupyter-notebook)
- [Using the Pre-trained Model](#using-the-pre-trained-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to develop a model capable of accurately detecting oil spills from aerial imagery. The DeepLabV3+ model, known for its effectiveness in semantic segmentation tasks, is utilized here with a ResNet101 backbone for feature extraction.

## Dataset

The `dataset.zip` file contains a collection of aerial images labeled for oil spills. The dataset is split into training, validation, and testing sets. Each image is accompanied by a corresponding mask where oil spills are marked in a distinct color.

## Model Architecture

The model architecture consists of:
- **ResNet101 Backbone**: Used for feature extraction from input images.
- **Atrous Spatial Pyramid Pooling (ASPP)**: Captures multi-scale context.
- **Decoder Module**: Refines the segmentation output for precise oil spill detection.

## Setup

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/oil-spill-detection.git
   cd oil-spill-detection
   ```

2. Extract the dataset:
   ```bash
   unzip dataset.zip
   ```

3. Ensure you have the necessary Python packages by creating a virtual environment and installing the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Running the Jupyter Notebook

The included Jupyter Notebook provides an interactive way to understand the dataset, visualize the model's performance, and experiment with different parameters. To run the notebook:

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the notebook file and follow the instructions.

## Using the Pre-trained Model

The `model.pth` file contains a pre-trained model that can be used to make predictions on new images. To use the model:

1. Load the model in a Python script:
   ```python
   import torch
   model = torch.load('model.pth')
   model.eval()
   ```

2. Preprocess your image and make predictions:
   ```python
   # Assuming `image` is a PIL Image object and `transform` is defined as in the notebook
   input_tensor = transform(image).unsqueeze(0)
   with torch.no_grad():
       output = model(input_tensor)
   ```

## Results

The model achieved a test accuracy of 69.73% and demonstrated its ability to correctly identify oil spills in live testing scenarios.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to reach out if you have any questions or suggestions. Star this repository if you find it helpful! 🌟
