# Oil Spill Detection using DeepLabV3+ Model

[![GitHub stars](https://img.shields.io/github/stars/rsrsp21/repo?style=social)](https://github.com/rsrsp21/repo)
[![GitHub forks](https://img.shields.io/github/forks/rsrsp21/repo?style=social)](https://github.com/rsrsp21/repo)

This repository contains the implementation of an oil spill detection system using the DeepLabV3+ model with a ResNet101 backbone. The project includes a Jupyter Notebook for exploration, a dataset of labeled images, and a pre-trained model for making predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Running the Jupyter Notebook](#running-the-jupyter-notebook)
- [Using the Pre-trained Model](#using-the-pre-trained-model)
- [Loading the Model with Auxiliary Classifier](#loading-the-model-with-auxiliary-classifier)
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

## Loading the Pre-trained Model with Auxiliary Classifier

To leverage the capabilities of the DeepLabV3+ model with an auxiliary classifier for oil spill detection, follow these steps to load the pre-trained model:

1. **Instantiate the Model**: Begin by defining the DeepLabV3+ model structure with a ResNet101 backbone and an auxiliary loss component. This auxiliary classifier enhances the learning process by providing additional gradient signals.

```python
import torch
import torchvision.models.segmentation as segmentation_models

# Define the model with auxiliary classifier
model = segmentation_models.deeplabv3_resnet101(weights=None, aux_loss=True)
model.classifier = segmentation_models.DeepLabHead(2048, 4)
```

2. **Prepare the Device**: Determine whether a CUDA-enabled GPU is available for training. If a GPU is available, the model will be loaded onto the GPU; otherwise, it will default to the CPU.

```python
# Prepare the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

3. **Load the State Dictionary**: Load the pre-trained weights into the model structure. Ensure that the state dictionary is compatible with the model architecture.

```python
# Load the saved state dictionary
model.load_state_dict(torch.load('deeplabv3_resnet101_oil_spill.pth', map_location=device))
```

4. **Transfer to Device**: Move the model to the prepared device (GPU or CPU).

```python
# Transfer the model to the device
model = model.to(device)
```

5. **Set Evaluation Mode**: Before making predictions or evaluating the model, set it to evaluation mode to disable dropout layers and batch normalization.

```python
# Set the model to evaluation mode
model.eval()
```

After completing these steps, the model is ready to process new images and predict oil spills. You can integrate this model into a larger application or use it as a standalone predictor.

```python
# Example of how to use the loaded model (assuming you have a preprocessed image tensor 'input_tensor')
with torch.no_grad():
    output = model(input_tensor.to(device))['out']
```

Make sure to replace `'deeplabv3_resnet101_oil_spill.pth'` with the actual path to your saved model file. Also, ensure that the input tensor `input_tensor` is preprocessed according to the model's requirements.

## Results

The model achieved a test accuracy of 69.73% and demonstrated its ability to correctly identify oil spills in live testing scenarios.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to reach out if you have any questions or suggestions. Star this repository if you find it helpful! 🌟
