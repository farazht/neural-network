/**
 * @file NeuralNetwork.c++
 * @brief This file contains the simple feedforward neural network class.
 */

#include "NeuralNetwork.h"
#include "LinearAlgebra.h"
#include <cmath>

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
        weights.push_back(Matrix(layers[i + 1], layers[i]));
        biases.push_back(Matrix(layers[i + 1], 1));
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
            result.at(i, j) = 1 / (1 + exp(-input.at(i, j)));   
        }
    }

    return result;
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
            result.at(i, j) = std::max(0.0, input.at(i, j));
        }
    }

    return result;
}

/**
 * @brief Applies the softmax activation function to a matrix.
 *
 * The softmax function is defined as e^x_i / sum(e^x_j) for all j in [0, n), 
 * where x is the input vector and n is the length of the vector.
 * 
 * @param input The matrix to apply the softmax function to.
 * @return The matrix with the softmax function applied to each element.
 */
Matrix NeuralNetwork::softmax(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());

    for (int j = 0; j < input.getCols(); j++) {
        double sum_exp = 0.0;
        
        for (int i = 0; i < input.getRows(); i++) {
            sum_exp += std::exp(input.at(i, j));
        }

        for (int i = 0; i < input.getRows(); i++) {
            result.at(i, j) = std::exp(input.at(i, j)) / sum_exp;
        }
    }
    
    return result;
}

/**
 * @brief This function performs the feedforward pass through the neural network.
 * 
 * @param input The input matrix.
 * @return The output matrix.
 */
Matrix NeuralNetwork::feedforward(const Matrix& input) {
    Matrix activation = input;

    for (int i = 0; i < weights.size(); i++) {
        activation = add(multiply(weights[i], activation), biases[i]);

        if (i < weights.size() - 1) {
            activation = ReLU(activation);
        } else {
            activation = softmax(activation);
        }
    }

    return activation;
}
