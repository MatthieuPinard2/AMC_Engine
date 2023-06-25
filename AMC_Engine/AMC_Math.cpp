#include "AMC_Math.h"

size_t Matrix::memorySize() const noexcept {
    return m_nRows * m_nCols * sizeof(double);
}

Matrix::Matrix() noexcept : m_nRows(0), m_nCols(0), m_data(nullptr) {}

Matrix::Matrix(const size_t nRows, const size_t nCols) : m_nRows(nRows), m_nCols(nCols) {
    m_data = static_cast<double*>(_aligned_malloc(memorySize(), 64));
}

Matrix::Matrix(Matrix const& other) : Matrix(other.m_nRows, other.m_nCols) {
    memcpy(m_data, other.m_data, memorySize());
}

Matrix::Matrix(Matrix&& other) noexcept {
    m_data = std::exchange(other.m_data, nullptr);
    m_nRows = std::exchange(other.m_nRows, 0);
    m_nCols = std::exchange(other.m_nCols, 0);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (&other != this) {
        const size_t thisSize = memorySize();
        m_nRows = other.m_nRows;
        m_nCols = other.m_nCols;
        const size_t otherSize = memorySize();
        if (otherSize > thisSize) {
            m_data = static_cast<double*>(_aligned_realloc(m_data, otherSize, 64));
        }
        memcpy(m_data, other.m_data, otherSize);
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (&other != this) {
        _aligned_free(m_data);
        m_data = std::exchange(other.m_data, nullptr);
        m_nRows = std::exchange(other.m_nRows, 0);
        m_nCols = std::exchange(other.m_nCols, 0);
    }
    return *this;
}

Matrix::~Matrix() {
    _aligned_free(m_data);
}

size_t Matrix::getNbCols() const noexcept {
    return m_nCols;
}

size_t Matrix::getNbRows() const noexcept {
    return m_nRows;
}

double* Matrix::data() noexcept {
    return m_data;
}

const double* Matrix::data() const noexcept {
    return m_data;
}

double* Matrix::operator[](const size_t i) noexcept {
    return m_data + (m_nCols * i);
}

const double* Matrix::operator[](const size_t i) const noexcept {
    return m_data + (m_nCols * i);
}

// Solves min_x((A*x - b)^0), stores the result in b.
void solveLinearRegression_SVD(Matrix const& A, std::vector<double>& b) {
    lapack_int rank = 0;
    auto s = static_cast<double*>(mkl_malloc(A.getNbRows() * sizeof(double), 64));
    Matrix A_copy = A;
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
void productMatrixVector(Matrix const& A, std::vector<double>& b) {
    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        static_cast<int>(A.getNbRows()),
        static_cast<int>(A.getNbCols()),
        1.0,
        A.data(),
        static_cast<int>(A.getNbCols()),
        std::vector(b).data(),
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

// Computes the mean and standard deviation of each column of a matrix X.
void standardDeviation(Matrix const& X, std::vector<double>& mean, std::vector<double>& std) {
    const size_t samples = X.getNbRows();
    const size_t cols = X.getNbCols();
    for (size_t j = 0; j < cols; ++j) {
        mean[j] = 0.0;
        std[j] = 0.0;
    }
    for (size_t i = 0; i < samples; ++i) {
        auto xRow = X[i];
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
