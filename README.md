# Enhanced Autonomous Driving: PrediNet20 with Adaptive Huber Loss

## Abstract

This repository hosts the implementation of the research study "Enhanced Autonomous Driving: PrediNet20 with Adaptive Huber Loss with Regularization for Improved Performance." The research introduces PrediNet20, a state-of-the-art Convolutional Neural Network (CNN), tailored for autonomous driving using the Donkeycar platform.

## PrediNet20 Model Definition

PrediNet20 is a specialized CNN developed to address the challenges of predicting throttle and steering angles in autonomous driving. The architecture consists of:

- **Convolutional Layers**: Designed to capture spatial hierarchies and patterns in the input images.
- **Fully Connected Layers**: Tailored to combine the features and make final predictions.
- **Activation Functions**: Utilizes activation functions such as ReLU for introducing non-linearities.

## Adaptive Huber Loss with Regularization (AHLR)

The AHLR is a novel loss function, a key contribution of this research. It is defined as follows:

- **Adaptive Nature**: Dynamically adjusts the loss based on the prediction error magnitude, transitioning from quadratic to linear loss.
- **Huber Component**: Controls the transition between quadratic and linear loss, minimizing the impact of outliers.
- **L1 Regularization**: Included to encourage sparsity in the model weights, aiding in reducing overfitting.

The combination of these elements leads to improved training convergence and model generalization.

## Experimental Setup

The repository includes code and resources for replicating the extensive experiments conducted with real driving data, demonstrating PrediNet20's superior accuracy and efficiency.

## Contents

- `src/`: Source code, including model architecture, loss function.
- `data/`: Real driving data used for training and validation.
- `models/`: Pre-trained models and weights.
- `results/`: Graphs, charts, and analyses of the experimental results.


## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
