/**
 * @file NeuralNetwork.c++
 * @brief This file contains the simple feedforward neural network class.
 */

#include "NeuralNetwork.h"
#include "LinearAlgebra.h"

/**
 * @brief Hyperparameters for the neural network - feel free to adjust.
 */
const double NeuralNetwork::LEARNING_RATE = 0.001;

/**
 * @brief Initializes the neural network with the given layers.
 * 
 * Weights control the strength of connections between neurons in adjacent layers.
 * Each weight matrix is of size [<neurons in next layer> x <neurons in current layer>].
 * 
 * Biases allow for the network to shift the activation function in the output layer.   
 * Each bias vector is of size [<neurons in next layer> x 1].
 * 
 * @param layers The number of neurons in each layer.
 */
NeuralNetwork::NeuralNetwork(const std::vector<int>& layers) : layers(layers) {
    for (int i = 0; i < layers.size() - 1; i++) {
        double limit = std::sqrt(2.0 / layers[i]);
        weights.push_back(Matrix::randomMatrix(layers[i + 1], layers[i], -limit, limit));
        biases.push_back(Matrix::randomMatrix(layers[i + 1], 1, -limit, limit));
    }
}

/**
 * @brief Applies the sigmoid activation function to a matrix.
 * 
 * The sigmoid function is defined as 1 / (1 + e^(-x)), where x is the input matrix.    
 * 
 * @param input The matrix to apply the sigmoid function to.
 * @return The matrix with the sigmoid function applied to each element.
 */
Matrix NeuralNetwork::sigmoid(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());

    for (int i = 0; i < input.getRows(); i++) {
        for (int j = 0; j < input.getCols(); j++) {
            result.at(i, j) = 1 / (1 + std::exp(-input.at(i, j)));   
        }
    }

    return result;
}

/**
 * @brief Applies the derivative of the sigmoid activation function to a matrix.
 *
 * The sigmoid derivative function is defined as sigmoid(x) * (1 - sigmoid(x)), 
 * where x is the input matrix.
 * 
 * @param input The matrix to apply the sigmoid derivative function to.
 * @return The matrix with the sigmoid derivative function applied to each element.
 */
Matrix NeuralNetwork::sigmoidDerivative(const Matrix& input) {
    Matrix sigmoid_result = sigmoid(input);
    Matrix ones = Matrix::valueMatrix(sigmoid_result.getRows(), sigmoid_result.getCols(), 1.0);
    return hadamardProduct(sigmoid_result, subtract(ones, sigmoid_result));
}

/**
 * @brief Applies the ReLU activation function to a matrix.   
 * 
 * The ReLU function is defined as max(0, x), where x is the input matrix.
 * 
 * @param input The matrix to apply the ReLU function to.
 * @return The matrix with the ReLU function applied to each element.
 */
Matrix NeuralNetwork::ReLU(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());

    for (int i = 0; i < input.getRows(); i++) {
        for (int j = 0; j < input.getCols(); j++) {
            result.at(i, j) = (input.at(i, j) > 0) ? input.at(i, j) : 0;
        }
    }

    return result;
}

/**     
 * @brief Applies the derivative of the ReLU activation function to a matrix.
 *
 * The ReLU derivative function is defined as 0 if x <= 0, and 1 otherwise.
 * 
 * @param input The matrix to apply the ReLU derivative function to.
 * @return The matrix with the ReLU derivative function applied to each element.
 */
Matrix NeuralNetwork::ReLUDerivative(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());

    for (int i = 0; i < input.getRows(); i++) {
        for (int j = 0; j < input.getCols(); j++) {
            result.at(i, j) = (input.at(i, j) > 0) ? 1 : 0;
        }
    }

    return result;
}

/**
 * @brief Applies the softmax activation function column-wise to a matrix.
 *
 * The softmax function is defined as e^x_i / sum(e^x_j) for all j in [0, n), 
 * where x is a column vector in the input matrix, and n is the number of rows.
 *
 * @param input The matrix where each column represents a vector of logits.
 * @return The matrix with the softmax function applied to each column.
 */
Matrix NeuralNetwork::softmax(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    
    for (int j = 0; j < input.getCols(); j++) {
        double sum_exp = 0.0;
        
        for (int i = 0; i < input.getRows(); i++) {
            result.at(i, j) = std::exp(input.at(i, j));
            sum_exp += result.at(i, j);
        }
        
        for (int i = 0; i < input.getRows(); i++) {
            result.at(i, j) /= sum_exp;
        }
    }
    
    return result;
}


/**
 * @brief This function performs the feedforward pass through the neural network.
 *
 * In the feedforward pass, we apply the ReLU activation function to the output
 * of each layer, until we reach the output layer, where we apply the softmax
 * activation function instead.
 * 
 * @param input The input matrix.
 * @return The output matrix.
 */
Matrix NeuralNetwork::feedforward(const Matrix& input) {
    Matrix activation = input;

    for (int i = 0; i < weights.size(); i++) {
        Matrix z = add(multiply(weights[i], activation), biases[i]);
        
        if (i < weights.size() - 1) {
            activation = ReLU(z);
        } else {
            activation = softmax(z);
        }
    }

    return activation;
}


/**
 * @brief This function performs the backpropagation pass through the neural network.
 *
 * In the backpropagation pass, we calculate the error between the expected output  
 * and the actual output, and then we update the weights and biases to minimize the
 * error.
 * 
 * @param input The input matrix.
 * @param expected The expected output matrix.
 */
void NeuralNetwork::backpropagate(const Matrix& input, const Matrix& expected) {
    std::vector<Matrix> activations = {input};
    std::vector<Matrix> zs;
    
    // modified feedforward pass which stores intermediate values
    Matrix activation = input;
    for (int i = 0; i < weights.size(); i++) {
        Matrix z = add(multiply(weights[i], activation), biases[i]);
        zs.push_back(z);
        
        if (i < weights.size() - 1) {
            activation = ReLU(z);
        } else {
            activation = softmax(z);
        }
        activations.push_back(activation);
    }
    
    Matrix delta = subtract(activations.back(), expected);
    
    // minimizes error by updating weights and biases
    for (int i = weights.size() - 1; i >= 0; i--) {
        Matrix weight_gradient = multiply(delta, transpose(activations[i]));
        
        if (i > 0) {
            Matrix weighted_error = multiply(transpose(weights[i]), delta);
            delta = hadamardProduct(weighted_error, ReLUDerivative(zs[i-1]));
        }
        
        weights[i] = subtract(weights[i], scalarMultiply(LEARNING_RATE, weight_gradient));
        biases[i] = subtract(biases[i], scalarMultiply(LEARNING_RATE, delta));
    }
}
