#include <iostream> // input/output to console
#include <vector> // dynamic array
#include <stdexcept> // exception handling

/**
 * @file LinearAlgebra.c++
 * @brief This file contains the matrix class along with some basic linear algebra operations.
 */
class Matrix {
    private:
        int rows;
        int cols;
        std::vector<std::vector<double>> data;

    public:
        Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0)) {}
        int getRows() const { return rows; }
        int getCols() const { return cols; }
        double& at(int row, int col) { return data[row][col]; }
};

/**
 * @brief This function adds two matrices.
 * 
 * Matrix addition:
 * [ a b ]   [ e f ]   [ a+e b+f ]
 * [ c d ] + [ g h ] = [ c+g d+h ]
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return The sum of the two matrices.
 */
Matrix add(const Matrix& a, const Matrix& b) {
    if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    Matrix result(a.getRows(), a.getCols());

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            result.at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }

    return result;
}

/**
 * @brief This function subtracts two matrices.
 * 
 * Matrix subtraction:
 * [ a b ]   [ e f ]   [ a-e b-f ]
 * [ c d ] - [ g h ] = [ c-g d-h ]
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return The difference of the two matrices.
 */
Matrix subtract(const Matrix& a, const Matrix& b) {
    if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    Matrix result(a.getRows(), a.getCols());

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            result.at(i, j) = a.at(i, j) - b.at(i, j);
        }
    }

    return result;
}

/**
 * @brief This function multiplies two matrices.
 * 
 * Matrix multiplication:
 * [ a b ]   [ e f ]   [ ae+bg af+bh ]
 * [ c d ] * [ g h ] = [ ce+dg cf+dh ]
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return The product of the two matrices.
 */
Matrix multiply(const Matrix& a, const Matrix& b) {
    if (a.getCols() != b.getRows()) {
        throw std::invalid_argument("Matrix a's column dimension must match matrix b's row dimension.");
    }

    Matrix result(a.getRows(), b.getCols());

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < b.getCols(); j++) {
            double sum = 0.0;

            for (int k = 0; k < a.getCols(); k++) {
                sum += a.at(i, k) * b.at(k, j);
            }

            result.at(i, j) = sum;
        }
    }

    return result;
}

/**
 * @brief This function multiplies a matrix by a scalar.
 * 
 * Scalar multiplication:
 * [ a b ]       [ ka kb ]
 * [ c d ] * k = [ kc kd ]
 * 
 * @param scalar The scalar.
 * @param matrix The matrix.
 * @return The product of the scalar and the matrix.
 */
Matrix scalarMultiply(double scalar, const Matrix& matrix) {
    Matrix result(matrix.getRows(), matrix.getCols());

    for (int i = 0; i < matrix.getRows(); i++) {
        for (int j = 0; j < matrix.getCols(); j++) {
            result.at(i, j) = scalar * matrix.at(i, j);
        }
    }

    return result;
}