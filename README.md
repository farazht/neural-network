# 1. Overview

This project contains the implementation of an artificial neural network, built using only C++ with no libraries other than the standard ones for input/output and file handling.

The code, primarily in `NeuralNetwork.c++`, `LinearAlgebra.c++`, and `Main.c++`, is well-documented and should be easy to follow and learn from.

The neural network is applicable to most classification problems. For testing purposes, the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is available, which scores around **97% accuracy** with the hyperparameters provided in `Main.c++` and `NeuralNetwork.c++`.

# 2. Setup

### 2.1. Clone the repository

HTTPS:

```bash
git clone https://github.com/farazht/neural-network.git
```

SSH:

```bash
git clone git@github.com:farazht/neural-network.git
```

### 2.2. Customization

You can easily customize the neural network in any of the following ways:
- Adjust `LEARNING_RATE` hyperparameter (in `NeuralNetwork.c++`, line 12)
- Adjust number and size of hidden layers (modify input vector in `Main.c++`, line 75)
- Adjust `EPOCHS` hyperparameter (in `Main.c++`, line 86)
- Change activation function (replace `ReLU` with `sigmoid` and `ReLUDerivative` with `sigmoidDerivative`, or vice versa)
- Change initialization strategy (modify `INITIALIZATION_STRATEGY` in `NeuralNetwork.c++`, line 13) 
    - 1 = He
    - 2 = Xavier/Glorot
    - 3 = LeCun
- Choose a dataset of your choice, as long as it is in the same format as the provided `mnist_train.txt` and `mnist_test.txt` 
    - Comma-separated values which normalize to between 0 and 1 (currently being normalized in `Main.c++`)
    - Label at end of line

### 2.3. Compilation

Install `g++` if it is not already installed, then run the following command in the project's root directory:

```bash
g++ -o Main LinearAlgebra.c++ NeuralNetwork.c++ Main.c++ -std=c++11
```

### 2.4. Execution

Run the compiled program:

```bash
./Main
```

# 3. How it works

### 3.1. Prerequisite: Linear Algebra

This project contains a custom implementation of a linear algebra library, which is used to handle multiple matrix operations required for the neural network.

The `Matrix` class provides the following operations:

*Initialization:*
- `Matrix(rows, cols)`: Creates empty matrix 
- `randomMatrix(rows, cols, min, max)`: Creates matrix filled with random values between min and max
- `valueMatrix(rows, cols, value)`: Creates matrix filled with given value

*Operations:*
- `add(Matrix a, Matrix b)`: Adds two matrices
- `subtract(Matrix a, Matrix b)`: Subtracts two matrices
- `multiply(Matrix a, Matrix b)`: Multiplies two matrices
- `transpose(Matrix m)`: Transposes a matrix
- `scalarMultiply(double scalar, Matrix m)`: Multiplies a matrix by a scalar
- `hadamardProduct(Matrix a, Matrix b)`: Takes the Hadamard product (a.k.a. element-wise product) of two matrices

*Information:*
- `at(row, col)`: Accesses element at given position
- `getRows()`: Gets number of rows
- `getCols()`: Gets number of columns

### 3.2. Neural Network

Using the linear algebra library, the neural network performs the following operations:

*Feedforward:*
1. For each layer, computes z = Wx + b
   - Uses `multiply()` to perform matrix multiplication of weights (W) and input (x)
   - Uses `add()` to add the bias vector (b)
2. Applies activation function to z
   - ReLU or sigmoid for hidden layers
   - Softmax for output layer
3. Passes result as input to next layer
   - Final output is probability distribution across classes

*Backpropagation:*
1. Calculates output error (y - ŷ)
   - Uses `subtract()` to find difference between expected (y) and actual (ŷ) output
2. For each layer from last to first:
   - Computes weight gradients:
     - Uses `multiply()` with error and transposed activation
     - Uses `transpose()` to get correct dimensions
   - Updates weights and biases:
     - Uses `scalarMultiply()` to apply learning rate
     - Uses `subtract()` to adjust parameters
   - Propagates error backward:
     - Uses `multiply()` with transposed weights and current error
     - Uses `hadamardProduct()` with activation derivative