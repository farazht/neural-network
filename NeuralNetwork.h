/**
 * @file NeuralNetwork.h
 * @brief File header for the NeuralNetwork.c++ file.
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "LinearAlgebra.h"

class NeuralNetwork {
private:
    std::vector<int> layers;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

public:
    NeuralNetwork(const std::vector<int>& layers);
    Matrix feedforward(const Matrix& input);
    static Matrix sigmoid(const Matrix& input);
    static Matrix ReLU(const Matrix& input);
    static Matrix softmax(const Matrix& input);
};

#endif
