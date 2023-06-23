#include <vector>

using Time = int;

class AMCFlow {
private:
    size_t m_exerciseIndex;
    Time m_observationDate;
    Time m_settlementDate;
    std::vector<double> m_amount;
public:
    inline size_t getExerciseIndex() const {
        return m_exerciseIndex;
    }
    inline Time getObservationDate() const {
        return m_observationDate;
    }
    inline Time getSettlementDate() const {
        return m_settlementDate;
    }
    inline double const& operator[](const size_t i) const {
        return m_amount[i];
    }
    inline double& operator[](const size_t i) {
        return m_amount[i];
    }
    AMCFlow(const size_t nPaths, const size_t exerciseIndex, const Time observationDate, const Time settlementDate) :
        m_exerciseIndex(exerciseIndex),
        m_observationDate(observationDate),
        m_settlementDate(settlementDate) {
        m_amount.resize(nPaths, 0.0);
    }
};

template <class T>
class Matrix {
private:
    size_t m_nRows, m_nCols;
    T* m_data;
public:
    Matrix(const size_t nRows, const size_t nCols) : m_nRows(nRows), m_nCols(nCols) {
        m_data = new T[m_nRows * m_nCols];
    }
    ~Matrix() {
        delete[](m_data);
    }
    inline size_t getNbCols() const {
        return m_nCols;
    }
    inline T* getRow(const size_t i) {
        return m_data + (m_nCols * i);
    }
    inline const T* getRow(const size_t i) const {
        return m_data + (m_nCols * i);
    }
    inline T& getElement(const size_t i, const size_t j) {
        return *(m_data + (m_nCols * i) + j);
    }
    inline T const& getElement(const size_t i, const size_t j) const {
        return *(m_data + (m_nCols * i) + j);
    }
};

class AMCExercise {
public:
    virtual bool isCallable() const {
        return false;
    }
    virtual bool isPutable() const {
        return false;
    }
    virtual void computeWeights(std::vector<double>& weights) const {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 1.0;
        }
    }
    virtual inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& conditionalExpectation,
        std::vector<double> const& performances) const = 0;
};

/* Callable Exercise */
class AMCExercise_Callable : protected AMCExercise {
public:
    virtual bool isCallable() const {
        return true;
    }
    inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& conditionalExpectation,
        std::vector<double> const& performances) const
    {
        for (size_t i = 0; i < exercise.size(); ++i) {
            exercise[i] = conditionalExpectation[i] >= 0.0 ? 1.0 : 0.0;
        }
    }
};

/* Putable Exercise */
class AMCExercise_Putable : protected AMCExercise {
public:
    virtual bool isPutable() const {
        return true;
    }
    inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& conditionalExpectation,
        std::vector<double> const& performances) const
    {
        for (size_t i = 0; i < exercise.size(); ++i) {
            exercise[i] = conditionalExpectation[i] <= 0.0 ? 1.0 : 0.0;
        }
    }
};

/* Null Exercise */
class AMCExercise_NoExercise : protected AMCExercise {
public:
    inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& conditionalExpectation,
        std::vector<double> const& performances) const
    {
        for (size_t i = 0; i < exercise.size(); ++i) {
            exercise[i] = 0.0;
        }
    }
};

// TODO : Discount factors.
double getDiscountFactor(Time end, Time begin, size_t iPath) {
    // Past flow.
    if (end < begin)
        return 0.0;
    // Future flow.
    return exp(-0.02 * (end - begin) / 365.0);
}

class AMCEngine {
private:
	// Flows of the European Contract
    std::vector<AMCFlow> m_contractFlows;
    std::vector<AMCFlow> m_exerciseFlows;
    std::vector<AMCExercise> m_exercises;
    size_t m_nExercises;
    size_t m_nPaths;
    bool m_useCrossTerms;
    Time m_modelDate;
    std::vector<double> m_conditionalExpectation;
    std::vector<double> m_premiumAfter;
    std::vector<double> m_premiumBefore;
    std::vector<double> m_exerciseDecision;
    std::vector<double> m_weights;
    Matrix<double> m_basis;
    size_t m_polynomialDegree;
    std::vector<Matrix<double>> m_stateVariables;
    std::vector<Matrix<double>> m_linearStateVariables;
    void updatePremiumBefore(const size_t exIdx) {
        Time exSettlementDate = m_exerciseFlows[exIdx].getSettlementDate();
        if (exIdx + 1 < m_nExercises) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_premiumBefore[j] = m_premiumAfter[j] * getDiscountFactor(m_exerciseFlows[exIdx + 1].getSettlementDate(), exSettlementDate, j);
            }
        }
        for (const auto& flow : m_contractFlows) {
            if (flow.getExerciseIndex() == exIdx + 1) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    m_premiumBefore[j] += flow[j] * getDiscountFactor(flow.getSettlementDate(), exSettlementDate, j);
                }
            }
        }
    }
    void rescaleStateVariable(Matrix<double>& svMatrix) const {
        const size_t nStateVariables = svMatrix.getNbCols();
        std::vector<double> x1(nStateVariables, 0.0), x2(nStateVariables, 0.0);
        for (size_t i = 0; i < m_nPaths; ++i) {
            for (size_t j = 0; j < nStateVariables; ++j) {
                const double SV = svMatrix.getElement(i, j);
                x1[j] += SV; x2[j] += SV * SV;
            }
        }
        for (size_t j = 0; j < nStateVariables; ++j) {
            x1[j] /= m_nPaths;
            x2[j] /= m_nPaths;
            x2[j] -= (x1[j] * x1[j]);
            x2[j] = sqrt(x2[j]);
        }
        for (size_t i = 0; i < m_nPaths; ++i) {
            for (size_t j = 0; j < nStateVariables; ++j) {
                svMatrix.getElement(i, j) -= x1[j];
                svMatrix.getElement(i, j) /= x2[j];
            }
        }
    }
    void computeBasis(Matrix<double>& svMatrix, Matrix<double>& linearSVMatrix) {
        const size_t nStateVariables = svMatrix.getNbCols();
        size_t basisSize;
        if (!m_useCrossTerms || nStateVariables <= 1) {
            for (size_t i = 0; i < m_nPaths; ++i) {
                m_basis.getElement(i, 0) = 1.0;
                for (size_t j = 0; j < nStateVariables; ++j) {
                    const double x = svMatrix.getElement(i, j);
                    double x_k = x;
                    size_t offset = j * m_polynomialDegree;
                    for (size_t k = 1; k <= m_polynomialDegree; ++k) {
                        m_basis.getElement(i, offset + k) = x_k;
                        x_k *= x;
                    }
                }
            }
            basisSize = 1 + nStateVariables * m_polynomialDegree;
        } else {
            // Not implemented.
            basisSize = 0;
        }
        const size_t nLinearStateVariables = linearSVMatrix.getNbCols();
        for (size_t i = 0; i < m_nPaths; ++i) {
            for (size_t j = 0; j < nLinearStateVariables; ++j) {
                const double x = linearSVMatrix.getElement(i, j);
                size_t offset = (j + 1) * basisSize;
                for (size_t k = 0; k < basisSize; ++k) {
                    m_basis.getElement(i, offset + k) = m_basis.getElement(i, k) * x;
                }
            }
        }
    }
    void clampConditionalExpectation(const size_t exIdx) {
        double minimumGain = DBL_MAX, maximumGain = -DBL_MAX;
        for (size_t j = 0; j < m_nPaths; ++j) {
            const double gain = m_premiumBefore[j] - m_exerciseFlows[exIdx][j];
            minimumGain = std::min(minimumGain, gain);
            maximumGain = std::max(maximumGain, gain);
        }
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_conditionalExpectation[j] = std::max(minimumGain, std::min(m_conditionalExpectation[j], maximumGain));
        }
    }
    void updateFutureFlows(const size_t exIdx) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_exerciseFlows[exIdx][j] *= m_exerciseDecision[j];
        }
        for (size_t nextEx = exIdx + 1; nextEx < m_nExercises; ++nextEx) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_exerciseFlows[nextEx][j] *= (1.0 - m_exerciseDecision[j]);
            }
        }
        for (auto& flow : m_contractFlows) {
            // Flows with Exercise Index == (m_nExercises + 1) are bullet (paid regardless of an exercise event)
            if (flow.getExerciseIndex() > exIdx && flow.getExerciseIndex() != m_nExercises + 1) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    flow[j] *= (1.0 - m_exerciseDecision[j]);
                }
            }
        }
    }
public:
    AMCEngine() {
        // m_exercises = the exercise definitions.
        m_nExercises = m_exercises.size();
        // m_nPaths = the number of paths.
        // m_modelDate = the model date.
        // m_polynomialDegree
        // m_useCrossTerms
        m_conditionalExpectation.resize(m_nPaths, 0.0);
        m_premiumAfter.resize(m_nPaths, 0.0);
        m_premiumBefore.resize(m_nPaths, 0.0);
        m_exerciseDecision.resize(m_nPaths, 0.0);
    }
	void computeForward() {
        // m_contractFlows = the contract flows. (the continuation flows)
        // m_exerciseFlows = the exercise flows. (the rebates)
        // Don't forget to set the exercise index of the contract flows and exercise flows.
        // Exercise Index of a flow = i <=> obs(i-1) < obs(flow) <= obs(i)
	}
    void computeBackward() {
        // The backward loop.
        size_t exIdx = m_nExercises - 1;
        while (exIdx < m_nExercises && m_exerciseFlows[exIdx].getObservationDate() >= m_modelDate) {
            updatePremiumBefore(exIdx);
            // Rescale state variables
            rescaleStateVariable(m_stateVariables[exIdx]);
            rescaleStateVariable(m_linearStateVariables[exIdx]);
            // Basis computation
            computeBasis();
            // m_conditionalExpectation is the conditional expectation of (m_premiumBefore - rebate).
            // m_conditionalExpectation = 
            
            clampConditionalExpectation(exIdx);
            /* Compute the exercise decision */
            m_exercises[exIdx].computeExercise(m_exerciseDecision, m_conditionalExpectation, m_performances);
            const bool isCallableOrPutable = m_exercises[exIdx].isCallable() || m_exercises[exIdx].isPutable();
            double premiumBeforeEx = 0.0, premiumAfterEx = 0.0;
            if (isCallableOrPutable) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    premiumBeforeEx += m_premiumBefore[j];
                }
            }
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_premiumAfter[j] = m_exerciseDecision[j] * m_exerciseFlows[exIdx][j] + (1.0 - m_exerciseDecision[j]) * m_premiumBefore[j];
            }
            if (isCallableOrPutable) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    premiumAfterEx += m_premiumAfter[j];
                }
            }
            if ((m_exercises[exIdx].isCallable() && premiumAfterEx > premiumBeforeEx) ||
                (m_exercises[exIdx].isPutable()  && premiumAfterEx < premiumBeforeEx)) {
                // We made things worse : revert the exercise decision.
                for (size_t j = 0; j < m_nPaths; ++j) {
                    m_exerciseDecision[j] = 1.0 - m_exerciseDecision[j];
                }
                for (size_t j = 0; j < m_nPaths; ++j) {
                    m_premiumAfter[j] = m_exerciseDecision[j] * m_exerciseFlows[exIdx][j] + (1.0 - m_exerciseDecision[j]) * m_premiumBefore[j];
                }
            }
            updateFutureFlows(exIdx);
            exIdx--;
        }
        // We end up with the flows being in m_contractFlows and m_exerciseFlows (those ones are set to 0 !)
    }
};

