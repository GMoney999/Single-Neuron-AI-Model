# Simple Machine Learning Model: Celsius to Fahrenheit Conversion

## Overview
This project uses a basic neural network model implemented in TensorFlow and Keras to predict Fahrenheit temperatures from Celsius inputs. The goal is to demonstrate how a simple linear regression task can be approached using machine learning techniques, specifically a neural network with a single neuron.

## Requirements
- TensorFlow
- Keras
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Dataset
The model is trained on a dataset containing paired values of temperatures in Celsius and their corresponding values in Fahrenheit. The data is stored in a CSV file named `Celsius+to+Fahrenheit.csv`.

## Model Description
The model consists of a single dense layer with one neuron. The input to this neuron is the temperature in Celsius, and it outputs the corresponding temperature in Fahrenheit. The neuron's output is calculated as a linear combination of the input and a bias term, passed through a linear activation function (identity function).

### Model Configuration
- **Neurons**: 1
- **Activation**: Linear
- **Optimizer**: Adam (Learning rate: 0.8)
- **Loss Function**: Mean Squared Error

## Training
The model is trained for 300 epochs, and the training process involves optimizing the weights to minimize the mean squared error between the predicted and true Fahrenheit values.

## Visualization
During training, the loss is plotted against epochs to visualize the model's learning progress. This helps in understanding how quickly the model is converging towards the optimal weights.

## Evaluation and Prediction
After training, the model's performance can be evaluated by visualizing the loss reduction across epochs. Predictions can be made by inputting new Celsius values into the trained model.

## Usage
1. Import the dataset using Pandas.
2. Visualize the dataset using Seaborn to plot a scatterplot of Celsius vs. Fahrenheit.
3. Split the data into training sets.
4. Define and compile the model.
5. Train the model and visualize the training process.
6. Evaluate the model's performance and make predictions.

## Example Prediction
After training, you can test the model with a new Celsius value, like 10 degrees, and it will predict the Fahrenheit equivalent.

## How to Run
Ensure you have all required libraries installed, then run the script through a Python environment capable of handling machine learning workloads, such as Jupyter Notebook or a standard Python script executor.

