/**
 * @file LinearAlgebra.c++
 * @brief This file contains some basic linear algebra operations for the Matrix class.
 */

#include "LinearAlgebra.h"
#include <iostream>

/**
 * @brief Initializes a matrix with the given number of rows and columns.
 * 
 * @param rows Number of rows.
 * @param cols Number of columns.
 */     
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0)) {}

/**
 * @brief Initializes a matrix with random values between the given minimum and maximum.
 * 
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param min Minimum value.
 * @param max Maximum value.
 * @return The initialized matrix.
 */
Matrix Matrix::randomMatrix(int rows, int cols, double min, double max) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = min + (max - min) * rand() / RAND_MAX;
        }
    }
    return result;
}

/**
 * @brief Initializes a matrix with all elements set to the given value.
 * 
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param value Value to set all elements to.
 * @return The initialized matrix.
 */
Matrix Matrix::valueMatrix(int rows, int cols, double value) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = value;
        }
    }
    return result;
}

/**
 * @brief Returns the number of rows in the matrix.
 * 
 * @return Number of rows.
 */
int Matrix::getRows() const { 
    return rows; 
}

/**
 * @brief Returns the number of columns in the matrix.
 * 
 * @return Number of columns.
 */
int Matrix::getCols() const { 
    return cols; 
}

/**
 * @brief Returns the value at the given row and column.
 * 
 * @param row The row.
 * @param col The column.
 * @return Value at the given row and column.
 */
double Matrix::at(int row, int col) const { 
    return data[row][col]; 
}

/**
 * @brief Returns a pointer to the value at the given row and column.
 * 
 * @param row The row.
 * @param col The column.
 * @return Pointer to the value at the given row and column.
 */
double& Matrix::at(int row, int col) { 
    return data[row][col]; 
}

/**
 * @brief Adds two matrices.
 * 
 * Matrix addition:
 * [ a b ]   [ e f ]   [ a+e b+f ]
 * [ c d ] + [ g h ] = [ c+g d+h ]
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return Sum of the two matrices.
 */
Matrix add(const Matrix& a, const Matrix& b) {
    Matrix result(a.getRows(), a.getCols());

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            result.at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }

    return result;
}

/**
 * @brief Subtracts two matrices.
 * 
 * Matrix subtraction:
 * [ a b ]   [ e f ]   [ a-e b-f ]
 * [ c d ] - [ g h ] = [ c-g d-h ]
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return Difference of the two matrices.
 */
Matrix subtract(const Matrix& a, const Matrix& b) {
    Matrix result(a.getRows(), a.getCols());

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            result.at(i, j) = a.at(i, j) - b.at(i, j);
        }
    }

    return result;
}

/**
 * @brief Multiplies two matrices.
 * 
 * Matrix multiplication:
 * [ a b ]   [ e f ]   [ ae+bg af+bh ]
 * [ c d ] * [ g h ] = [ ce+dg cf+dh ]
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return Product of the two matrices.
 */
Matrix multiply(const Matrix& a, const Matrix& b) {
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
 * @brief Multiplies a matrix by a scalar.
 * 
 * Scalar multiplication:
 * [ a b ]       [ ka kb ]
 * [ c d ] * k = [ kc kd ]
 * 
 * @param scalar The scalar.
 * @param matrix The matrix.
 * @return Product of the scalar and the matrix.
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

/**
 * @brief Calculates the Hadamard product of two matrices.
 * 
 * Hadamard product, also known as elementwise multiplication:
 * [ a b ]   [ e f ]   [ ae be ]
 * [ c d ] * [ g h ] = [ ce de ]
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return Hadamard product of the two matrices.
 */
Matrix hadamardProduct(const Matrix& a, const Matrix& b) {
    Matrix result(a.getRows(), a.getCols());

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            result.at(i, j) = a.at(i, j) * b.at(i, j);
        }
    }

    return result;
}

/**
 * @brief Transposes a matrix.
 * 
 * Matrix transposition:
 * [ a b ]    [ a c ]
 * [ c d ] -> [ b d ]
 * 
 * @param matrix The matrix to transpose.
 * @return The transposed matrix.
 */
Matrix transpose(const Matrix& matrix) {
    Matrix result(matrix.getCols(), matrix.getRows());

    for (int i = 0; i < matrix.getRows(); i++) {
        for (int j = 0; j < matrix.getCols(); j++) {
            result.at(j, i) = matrix.at(i, j);
        }
    }

    return result;
}

/**
 * @brief Calculates the mean squared error between two matrices.
 * 
 * Mean squared error:
 * 1/2 * (a - b)^2
 * 
 * @param a The first matrix.
 * @param b The second matrix.
 * @return Mean squared error between the two matrices.
 */
double meanSquaredError(const Matrix& a, const Matrix& b) {
    double error = 0.0;

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            double diff = a.at(i, j) - b.at(i, j);
            error += diff * diff;
        }
    }

    return error / (2.0 * a.getRows() * a.getCols());
}

/**
 * @brief Prints a matrix to the console.
 * 
 * @param matrix The matrix to print.
 */
void printMatrix(const Matrix& matrix) {
    for (int i = 0; i < matrix.getRows(); i++) {
        for (int j = 0; j < matrix.getCols(); j++) {
            std::cout << matrix.at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}