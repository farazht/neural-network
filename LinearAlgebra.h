/**
 * @file LinearAlgebra.h
 * @brief File header for the LinearAlgebra.c++ file.
 */
#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <vector>
#include <iostream>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

public:
    Matrix(int rows, int cols);
    static Matrix randomMatrix(int rows, int cols, double min, double max);
    int getRows() const;
    int getCols() const;
    double at(int row, int col) const;
    double& at(int row, int col);
};

Matrix add(const Matrix& a, const Matrix& b);
Matrix subtract(const Matrix& a, const Matrix& b);
Matrix multiply(const Matrix& a, const Matrix& b);
Matrix scalarMultiply(double scalar, const Matrix& matrix);
Matrix elementwiseMultiply(const Matrix& a, const Matrix& b);
double meanSquaredError(const Matrix& a, const Matrix& b);
Matrix transpose(const Matrix& matrix);
void printMatrix(const Matrix& matrix);

#endif