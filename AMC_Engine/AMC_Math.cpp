#include "AMC_Math.h"
#include <cassert>

// Solves min_x((A*x - b)^0), stores the result in b.
void solveLinearRegression_SVD(Matrix<double> const& A, std::vector<double>& b) {
    lapack_int rank = 0;
    auto s = static_cast<double*>(mkl_malloc(A.getNbRows() * sizeof(double), 64));
    Matrix<double> A_copy = A;
    LAPACKE_dgelsd(
        LAPACK_ROW_MAJOR,
        static_cast<int>(A.getNbRows()),
        static_cast<int>(A.getNbCols()),
        1,
        A_copy.data(),
        static_cast<int>(A.getNbCols()),
        b.data(),
        1,
        s,
        0.0,
        &rank
    );
    mkl_free(s);
}

// Computes b <- A * b.
void productMatrixVector(Matrix<double> const& A, std::vector<double>& b) {
    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        static_cast<int>(A.getNbRows()),
        static_cast<int>(A.getNbCols()),
        1.0,
        A.data(),
        static_cast<int>(A.getNbCols()),
        std::vector<double>(b).data(),
        1,
        0.0,
        b.data(),
        1
    );
}

// Computes the mean and standard deviation of a vector X.
void standardDeviation(std::vector<double> const& X, double& mean, double& std) {
    mean = 0.0;
    std = 0.0;
    const size_t samples = X.size();
    assert(samples);
    for (size_t i = 0; i < samples; ++i) {
        const double x = X[i];
        mean += x;
        std += x * x;
    }
    mean /= static_cast<double>(samples);
    std /= static_cast<double>(samples);
    std -= mean * mean;
    std = sqrt(std::max(0.0, std));
}

// Computes the weighted mean and standard deviation of a vector X.
void standardDeviation(std::vector<double> const& X, std::vector<double> const& W, double& mean, double& std) {
    mean = 0.0;
    std = 0.0;
    double sumW = 0.0;
    const size_t samples = X.size();
    for (size_t i = 0; i < samples; ++i) {
        const double x = X[i];
        const double w = W[i];
        mean += w * x;
        std += w * x * x;
        sumW += w;
    }
    if (sumW <= 0.0) {
        mean = 0.0;
        std = 0.0;
    }
    else {
        mean /= sumW;
        std /= sumW;
        std -= mean * mean;
        std = sqrt(std::max(0.0, std));
    }
}

// Computes the mean and standard deviation of each column of a matrix X.
void standardDeviation(Matrix<double> const& X, std::vector<double>& mean, std::vector<double>& std) {
    const size_t samples = X.getNbRows();
    assert(samples);
    const size_t cols = X.getNbCols();
    for (size_t j = 0; j < cols; ++j) {
        mean[j] = 0.0;
        std[j] = 0.0;
    }
    for (size_t i = 0; i < samples; ++i) {
        auto xRow = X[i];
        assert(xRow);
        for (size_t j = 0; j < cols; ++j) {
            const double x = xRow[j];
            mean[j] += x;
            std[j] += x * x;
        }
    }
    for (size_t j = 0; j < cols; ++j) {
        mean[j] /= static_cast<double>(samples);
        std[j] /= static_cast<double>(samples);
        std[j] -= (mean[j] * mean[j]);
        std[j] = sqrt(std::max(0.0, std[j]));
    }
}
