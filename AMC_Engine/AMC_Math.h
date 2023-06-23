#pragma once

#include <vector>
#include <mkl.h>

class Matrix {
private:
    size_t m_nRows;
    size_t m_nCols;
    double* m_data;
    inline size_t memorySize() const noexcept;
public:
    Matrix() noexcept;
    Matrix(const size_t nRows, const size_t nCols);
    ~Matrix();
    Matrix(Matrix const& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix&) = delete;
    inline size_t getNbCols() const noexcept;
    inline size_t getNbRows() const noexcept;
    inline double* data() noexcept;
    inline const double* data() const noexcept;
    inline double* operator[](const size_t i) noexcept;
    inline const double* operator[](const size_t i) const noexcept;
};

// Solves min_x((A*x - b)^0), stores the result in b.
void solveLinearRegression_SVD(Matrix const& A, std::vector<double>& b);

// Computes b <- A * b.
void productMatrixVector(Matrix const& A, std::vector<double>& b);

// Computes the mean and standard deviation of a vector X.
void standardDeviation(std::vector<double> const& X, double& mean, double& std);

// Computes the mean and standard deviation of each column of a matrix X.
void standardDeviation(Matrix const& X, std::vector<double>& mean, std::vector<double>& std);
