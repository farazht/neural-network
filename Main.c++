#include "NeuralNetwork.h"
#include "LinearAlgebra.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

/**
 * @brief Preprocess a single line of MNIST data
 *
 * Our dataset is already in the format of a text file with each line containing
 * comma separated values of pixel values (0-255) and a label for the digit (0-9).
 *
 * We create an input and expected matrix, which we can feed into the neural network.
 *
 * @param line The line to preprocess
 * @return A pair of matrices: the input and the expected output
 */
std::pair<Matrix, Matrix> preprocessLine(const std::string& line) {
    std::istringstream iss(line);
    std::string pixel;
    std::vector<double> pixels;
    
    while (std::getline(iss, pixel, ',')) {
        pixels.push_back(std::stod(pixel));
    }
    
    int label = pixels.back();
    pixels.pop_back();
        
    Matrix input(pixels.size(), 1);
    for (size_t i = 0; i < pixels.size(); ++i) {
        input.at(i, 0) = pixels[i] / 255.0;
    }
    
    Matrix expected(10, 1);
    for (int i = 0; i < 10; ++i) {
        expected.at(i, 0) = (i == label) ? 1.0 : 0.0;
    }
    
    return {input, expected};
}

/**
 * @brief Load dataset from file
 * @param filename The filename of the dataset
 * @return A vector of pairs of matrices: the input and the expected output
 */
std::vector<std::pair<Matrix, Matrix>> loadDataset(const std::string& filename) {
    std::vector<std::pair<Matrix, Matrix>> dataset;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Could not open file " << filename << std::endl;
        return dataset;
    }
    
    std::string line;
    int lineCount = 0;
    
    while (std::getline(file, line)) {
        dataset.push_back(preprocessLine(line));
        lineCount++;
        if (lineCount % 10000 == 0) {
            std::cout << "Processed " << lineCount << " lines from " << filename << std::endl;
        }
    }
    
    std::cout << "Finished loading dataset " << filename << std::endl;
    return dataset;
}

int main() {
    // 1. INITIALIZE NETWORK (784 inputs for 28x28 image, 10 outputs for 10 digits)
    // note: the number of neurons in the hidden layers can be adjusted
    NeuralNetwork nn({784, 64, 32, 10});
    
    // 2. LOAD TRAINING DATASET
    auto trainingData = loadDataset("mnist_train.txt");
    
    if (trainingData.empty()) {
        std::cout << "Error: No training data loaded" << std::endl;
        return 1;
    }
    
    // 3. TRAINING LOOP
    const int EPOCHS = 10;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double totalError = 0.0;
        int sampleCount = 0;
        
        std::cout << "Starting epoch " << epoch + 1 << "/" << EPOCHS << std::endl;
        for (const auto& [input, expected] : trainingData) {
            Matrix output = nn.feedforward(input);
            nn.backpropagate(input, expected);
            totalError += meanSquaredError(output, expected);
            
            sampleCount++;
            if (sampleCount % 1000 == 0) {
                std::cout << "Processed " << sampleCount << "/" << trainingData.size() << " samples in epoch " << epoch + 1 << std::endl;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << " Complete - Average Error: " << totalError / trainingData.size() << std::endl;
    }
    
    // 4. LOAD TEST DATASET
    auto testData = loadDataset("mnist_test.txt");
    
    if (testData.empty()) {
        std::cout << "Error: No test data loaded" << std::endl;
        return 1;
    }
    
    // 5. TESTING LOOP
    int correctPredictions = 0;
    int testCount = 0;
    
    for (const auto& [input, expected] : testData) {
        Matrix output = nn.feedforward(input);
        
        int predLabel = 0, trueLabel = 0;
        for (int i = 0; i < 10; ++i) {
            if (output.at(i, 0) > output.at(predLabel, 0)) predLabel = i;
            if (expected.at(i, 0) > expected.at(trueLabel, 0)) trueLabel = i;
        }
        
        if (predLabel == trueLabel) correctPredictions++;
        
        testCount++;
        if (testCount % 1000 == 0) {
            std::cout << "Tested " << testCount << "/" << testData.size() << " samples" << std::endl;
        }
    }
    
    // 6. FINAL RESULTS
    double accuracy = 100.0 * correctPredictions / testData.size();

    std::cout << "\nTesting complete. The neural network achieved an accuracy of " << accuracy << "%, correctly predicting " << correctPredictions << " out of " << testData.size() << " samples!" << std::endl;
    return 0;
}