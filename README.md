# Linear Function Detection

A web application for detecting linear functions using a simple neural network.

## Project Description

This application allows you to predict the outcome of a specified linear function using a simple neural network. You can define the parameters `a` and `b` of the linear function `y = ax + b`, then the application will train a neural network model to predict the outcome based on the input you provide.

## Features

- Define the parameters `a` and `b` of the linear function.
- Input values to be predicted by the model based on the specified linear function.
- Display training time, final loss, and final mean absolute error (MAE) after training.
- Show training metrics graph (loss and MAE) during training.

## How to Use

1. Clone this repository to your computer.
    ```bash
    git clone https://github.com/username/linear-function-detection.git
    ```
2. Navigate to the project directory.
    ```bash
    cd linear-function-detection
    ```
3. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit application.
    ```bash
    streamlit run app.py
    ```
5. Open the application in your browser and use the interface to input the linear function parameters and values you want to predict.

## Directory Structure


## Dependencies

- [Streamlit](https://streamlit.io/) - version 1.20.0
- [TensorFlow](https://www.tensorflow.org/) - version 2.14.0
- [NumPy](https://numpy.org/) - version 1.26.0
- [Matplotlib](https://matplotlib.org/) - version 3.8.0

## About

**Linear Function Detector** is a web application created to demonstrate the use of a simple neural network in predicting the outcome of a linear function.

### Application Goals:

- Teach the basic concepts of neural networks.
- Demonstrate how neural network models can be used for linear regression problems.

### How to Use:

1. Enter values for the parameters `a` and `b` of the linear function.
2. Input the value `x` to predict `y` using the specified linear function.
3. Click the "Predict" button to see the prediction results and training metrics.

### References:

- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Streamlit Documentation](https://docs.streamlit.io)
