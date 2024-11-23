/**
 * @file Test.c++
 * @brief Test the neural network.
 */

#include "NeuralNetwork.h"
#include "LinearAlgebra.h"
#include <iostream>

void printResults(const Matrix& input, const Matrix& output, const Matrix& expected) {
    std::cout << "\n - INPUT MATRIX - " << std::endl;
    printMatrix(input);
    
    std::cout << " - ACTUAL OUTPUT MATRIX - " << std::endl;
    printMatrix(output);
    
    std::cout << " - EXPECTED OUTPUT MATRIX - " << std::endl;
    printMatrix(expected);
    
    double error = meanSquaredError(output, expected);
    std::cout << "MSE: " << error << std::endl;
}

int main() {
    // 1. INITIALIZE NETWORK (12 inputs, 16 and 10 hidden neurons, 6 outputs)
    NeuralNetwork nn({12, 16, 10, 6});  

    // 2. CREATE INPUT MATRIX (12 inputs, 7 samples)
    Matrix input(12, 7);
    for (int j = 0; j < input.getCols(); j++) {
        for (int i = 0; i < input.getRows(); i++) {
            input.at(i, j) = (i + j) % 2;
        }
    }
    
    // 3. CREATE EXPECTED OUTPUT MATRIX (6 outputs, 7 samples)
    Matrix expected(6, 7);
    for (int j = 0; j < expected.getCols(); j++) {
        for (int i = 0; i < expected.getRows(); i++) {
            expected.at(i, j) = (i == j % expected.getRows()) ? 1.0 : 0.0;
        }
    }

    // 4. FEEDFORWARD PASS
    Matrix initial_output = nn.feedforward(input);
    if (initial_output.getRows() == 6 && initial_output.getCols() == 7){
        printResults(input, initial_output, expected);
    } else {
        std::cout << "incorrect dimensions" << std::endl;
    }
    
    // 5. BACKPROPAGATION LOOP
    const int epochs = 1000;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        nn.backpropagate(input, expected);
        
        // progress check
        if (epoch % 100 == 0) {
            std::cout << "\nEpoch " << epoch << ":" << std::endl;
            Matrix output = nn.feedforward(input);
            
            if (output.getRows() == 6 && output.getCols() == 7){
                printResults(input, output, expected);
            } else {
                std::cout << "incorrect dimensions" << std::endl;
            }
                        
            printResults(input, output, expected);
        }
    }
    
    return 0;
}