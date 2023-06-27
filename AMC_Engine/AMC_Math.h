#pragma once

#include <vector>
#include <mkl.h>

// Visual Studio does not implement the standard C11 aligned memory allocation.
#if defined _MSC_VER
#define aligned_malloc  _aligned_malloc
#define aligned_realloc _aligned_realloc
#define aligned_free    _aligned_free
#else // ^^^ Windows ^^^ // vvv Unix vvv //
#endif

template <class T>
class Matrix {
private:
    static constexpr size_t alignment = 64; // 512b
    size_t m_nRows;
    size_t m_nCols;
    T* m_data;
    size_t memorySize() const noexcept {
        return m_nRows * m_nCols * sizeof(T);
    }
public:
    Matrix() noexcept : m_nRows(0), m_nCols(0), m_data(nullptr) {}
    Matrix(const size_t nRows, const size_t nCols) : m_nRows(nRows), m_nCols(nCols) {
        m_data = static_cast<T*>(aligned_malloc(memorySize(), alignment));
    }
    ~Matrix() {
        aligned_free(m_data);
    }
    Matrix(Matrix const& other) : Matrix(other.m_nRows, other.m_nCols) {
        memcpy(m_data, other.m_data, memorySize());
    }
    Matrix(Matrix&& other) noexcept {
        m_data = std::exchange(other.m_data, nullptr);
        m_nRows = std::exchange(other.m_nRows, 0);
        m_nCols = std::exchange(other.m_nCols, 0);
    }
    Matrix& operator=(const Matrix& other) {
        if (&other != this) {
            const size_t thisSize = memorySize();
            const size_t otherSize = other.memorySize();
            if (otherSize > thisSize) {
                m_data = static_cast<T*>(aligned_realloc(m_data, otherSize, alignment));
                if (!m_data)
                    throw(std::bad_alloc());
            }
            memcpy(m_data, other.m_data, otherSize);
            m_nRows = other.m_nRows;
            m_nCols = other.m_nCols;
        }
        return *this;
    }
    Matrix& operator=(Matrix&& other) noexcept {
        if (&other != this) {
            aligned_free(m_data);
            m_data = std::exchange(other.m_data, nullptr);
            m_nRows = std::exchange(other.m_nRows, 0);
            m_nCols = std::exchange(other.m_nCols, 0);
        }
        return *this;
    }
    size_t getNbRows() const noexcept {
        return m_nRows;
    }
    size_t getNbCols() const noexcept {
        return m_nCols;
    }
    T* data() noexcept {
        return m_data;
    }
    const T* data() const noexcept {
        return m_data;
    }
    T* operator[](const size_t i) noexcept {
        return m_data + (m_nCols * i);
    }
    const T* operator[](const size_t i) const noexcept {
        return m_data + (m_nCols * i);
    }
};

// Solves min_x((A*x - b)^0), stores the result in b.
void solveLinearRegression_SVD(Matrix<double> const& A, std::vector<double>& b);

// Computes b <- A * b.
void productMatrixVector(Matrix<double> const& A, std::vector<double>& b);

// Computes the mean and standard deviation of a vector X.
void standardDeviation(std::vector<double> const& X, double& mean, double& std);

// Computes the mean and standard deviation of each column of a matrix X.
void standardDeviation(Matrix<double> const& X, std::vector<double>& mean, std::vector<double>& std);
