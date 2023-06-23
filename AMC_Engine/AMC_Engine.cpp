#include "AMC_Math.h"
#include "AMC_Flow.h"
#include "AMC_Exercise.h"

/*
1/ Cross terms 
2/ Clean AMC_Engine.cpp
3/ Smoothing of callable/putable exercise
4/ kMeans
5/ Rescale the basis or the SV ? Rescale linear SV ?
6/ Flow + expiry Flow exercise index assignment 
7/ Centered smoothing
8/ Exit proba, Exit Impact, Expected Maturity (and other indicators)
9/ Simultaneous Callable + Putable
10/ Conditional Callable / Putable
*/

class AMCEngine {
private:
    // Contract Flows (= flows paid when the contract is alive)
    std::vector<AMCFlow> m_contractFlows;
    // Rebate Flows (= flows paid when exercising)
    std::vector<AMCFlow> m_exerciseFlows;
    // Definition of Exercises.
    std::vector<AMCExercisePtr> m_exercises;
    size_t m_nExercises;
    // Engine Data
    size_t m_nPaths;
    size_t m_polynomialDegree;
    double m_exercisableProportion;
    Time m_modelDate;
    bool m_useCrossTerms;
    // Internal data for the Exercise computation.
    std::vector<double> m_conditionalExpectation;
    std::vector<double> m_premiumAfter;
    std::vector<double> m_premiumBefore;
    std::vector<double> m_exerciseDecision;
    std::vector<double> m_weights;
    Matrix m_basis;
    Matrix m_basisWeighted;
    //
    std::vector<Matrix> m_stateVariables;
    std::vector<Matrix> m_linearStateVariables;
    std::vector<Matrix> m_performances;
    /* Private functions */ 
    void rescaleStateVariable(Matrix& svMatrix) const {
        const size_t nStateVariables = svMatrix.getNbCols();
        std::vector<double> meanSV(nStateVariables), stdSV(nStateVariables);
        standardDeviation(svMatrix, meanSV, stdSV);
        for (size_t i = 0; i < m_nPaths; ++i) {
            auto svRow = svMatrix[i];
            for (size_t j = 0; j < nStateVariables; ++j) {
                svRow[j] -= meanSV[j];
                svRow[j] /= stdSV[j];
            }
        }
    }
    void computeBasis(Matrix& svMatrix, Matrix& linearSVMatrix) {
        const size_t nStateVariables = svMatrix.getNbCols();
        size_t basisSize;
        if (!m_useCrossTerms || nStateVariables <= 1) {
            for (size_t i = 0; i < m_nPaths; ++i) {
                auto basisRow = m_basis[i];
                auto svRow = svMatrix[i];
                basisRow[0] = 1.0;
                for (size_t j = 0; j < nStateVariables; ++j) {
                    const double x = svRow[j];
                    double x_k = x;
                    size_t offset = j * m_polynomialDegree;
                    for (size_t k = 1; k <= m_polynomialDegree; ++k) {
                        basisRow[offset + k] = x_k;
                        x_k *= x;
                    }
                }
            }
            basisSize = 1 + nStateVariables * m_polynomialDegree;
        }
        else {
            // Not implemented.
            basisSize = 0;
        }
        const size_t nLinearStateVariables = linearSVMatrix.getNbCols();
        for (size_t i = 0; i < m_nPaths; ++i) {
            auto linearSVRow = linearSVMatrix[i];
            auto basisRow = m_basis[i];
            for (size_t j = 0; j < nLinearStateVariables; ++j) {
                const double x = linearSVRow[j];
                size_t offset = (j + 1) * basisSize;
                for (size_t k = 0; k < basisSize; ++k) {
                    basisRow[offset + k] = basisRow[k] * x;
                }
            }
        }
    }
    void updateFutureFlows(const size_t exIdx) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_exerciseFlows[exIdx].scaleAmount(j, m_exerciseDecision[j]);
        }
        for (size_t nextEx = exIdx + 1; nextEx < m_nExercises; ++nextEx) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_exerciseFlows[nextEx].scaleAmount(j, 1.0 - m_exerciseDecision[j]);
            }
        }
        for (auto& flow : m_contractFlows) {
            // Flows with Exercise Index == (m_nExercises + 1) are bullet (paid regardless of an exercise event)
            if (flow.getExerciseIndex() > exIdx && flow.getExerciseIndex() != m_nExercises + 1) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    flow.scaleAmount(j, 1.0 - m_exerciseDecision[j]);
                }
            }
        }
    }
    void rescaleByWeights(const size_t exIdx) {
        m_exercises[exIdx]->computeWeights(m_weights, m_performances[exIdx]);
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_conditionalExpectation[j] *= m_weights[j];
        }
        const size_t basisSize = m_basis.getNbCols();
        for (size_t j = 0; j < m_nPaths; ++j) {
            auto basisWeightedRow = m_basisWeighted[j];
            auto basisRow = m_basis[j];
            const double w = m_weights[j];
            for (size_t k = 0; k < basisSize; ++k) {
                basisWeightedRow[k] = basisRow[k] * w;
            }
        }
    }
    bool needRegression(const size_t exIdx) {
        // We need to regress if there are some flows which are not known at the exercise *observation* date.
        const Time exDate = m_exerciseFlows[exIdx].getObservationDate();
        // knownDate is the date are which all flows are known.
        Time knownDate = exDate;
        for (const auto& flow : m_contractFlows) {
            if (flow.getExerciseIndex() > exIdx && flow.getExerciseIndex() != m_nExercises + 1) {
                knownDate = std::max(knownDate, flow.getObservationDate());
            }
        }
        for (size_t nextEx = exIdx + 1; nextEx < m_nExercises; ++nextEx) {
            // Exercise flows : discard null exercise.
            if (!dynamic_cast<AMCExercise_NoExercise*>(m_exercises[nextEx].get())) {
                knownDate = std::max(knownDate, m_exerciseFlows[nextEx].getObservationDate());
            }
        }
        return knownDate > exDate;
    }
    void clampConditionalExpectation(const double minimumGain, const double maximumGain) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_conditionalExpectation[j] = std::max(minimumGain, std::min(m_conditionalExpectation[j], maximumGain));
        }
    }
    void computeConditionalExpectation(const size_t exIdx) {
        // m_conditionalExpectation is the conditional expectation of (m_premiumBefore - rebate) - we regress on the exercise gain.
        double minimumGain = DBL_MAX, maximumGain = -DBL_MAX;
        for (size_t j = 0; j < m_nPaths; ++j) {
            const double gain = m_premiumBefore[j] - (m_exerciseFlows[exIdx].getAmount(j) * m_exerciseFlows[exIdx].getDFObsToSettleDate(j));
            m_conditionalExpectation[j] = gain;
            minimumGain = std::min(minimumGain, gain);
            maximumGain = std::max(maximumGain, gain);
        }
        // Check if we need to solve the linear regression - otherwise, just return.
        if (!needRegression(exIdx))
            return;
        // We rescale the trajectories by a weighting vector. This is done to give more importance to certain trajectories,
        // and discard other useless paths.
        rescaleByWeights(exIdx);
        // Solve the linear system using SVD. m_conditionalExpectation now contains the regressed values.
        solveLinearRegression_SVD(m_basisWeighted, m_conditionalExpectation);
        productMatrixVector(m_basis, m_conditionalExpectation);
        // For stability, we clamp the conditional expectation by the minimum and maximum realized values.
        clampConditionalExpectation(minimumGain, maximumGain);
    }
    void updatePremiumBefore(const size_t exIdx) {
        if (exIdx + 1 < m_nExercises) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_premiumBefore[j] = m_premiumAfter[j] * m_exerciseFlows[exIdx + 1].getDFToObservationDate(j) / m_exerciseFlows[exIdx].getDFToObservationDate(j);
            }
        }
        for (const auto& flow : m_contractFlows) {
            if (flow.getExerciseIndex() == exIdx + 1) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    m_premiumBefore[j] += flow.getAmount(j) * flow.getDFToSettlementDate(j) / m_exerciseFlows[exIdx].getDFToObservationDate(j);
                }
            }
        }
    }
    void updatePremiumAfter(const size_t exIdx) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_premiumAfter[j] = m_exerciseDecision[j] * (m_exerciseFlows[exIdx].getAmount(j) * m_exerciseFlows[exIdx].getDFObsToSettleDate(j)) + (1.0 - m_exerciseDecision[j]) * m_premiumBefore[j];
        }
        const bool isCallable = m_exercises[exIdx]->isCallable(), isPutable = m_exercises[exIdx]->isPutable();
        if (isCallable || isPutable) {
            double premiumBeforeEx = 0.0, premiumAfterEx = 0.0;
            for (size_t j = 0; j < m_nPaths; ++j) {
                premiumBeforeEx += m_premiumBefore[j];
                premiumAfterEx += m_premiumAfter[j];
            }
            premiumBeforeEx /= double(m_nPaths);
            premiumAfterEx /= double(m_nPaths);
            // We made things worse : revert the exercise decision.
            if ((isCallable && premiumAfterEx > premiumBeforeEx) || (isPutable && premiumAfterEx < premiumBeforeEx)) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    m_exerciseDecision[j] = 1.0 - m_exerciseDecision[j];
                    m_premiumAfter[j] = m_exerciseDecision[j] * (m_exerciseFlows[exIdx].getAmount(j) * m_exerciseFlows[exIdx].getDFObsToSettleDate(j)) + (1.0 - m_exerciseDecision[j]) * m_premiumBefore[j];
                }
            }
        }
        if (m_exercisableProportion < 1.0) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_exerciseDecision[j] = std::min(m_exercisableProportion, m_exerciseDecision[j]);
                m_premiumAfter[j] = m_exerciseDecision[j] * (m_exerciseFlows[exIdx].getAmount(j) * m_exerciseFlows[exIdx].getDFObsToSettleDate(j)) + (1.0 - m_exerciseDecision[j]) * m_premiumBefore[j];
            }
        }
    }
public:
    AMCEngine(AMCEngine const&) = delete;
    AMCEngine(AMCEngine &&) = delete;
    AMCEngine& operator=(AMCEngine const&) = delete;
    AMCEngine() {
        // m_exercises = the exercise definitions.
        m_nExercises = m_exercises.size();
        // m_nPaths = the number of paths.
        // m_modelDate = the model date.
        // m_polynomialDegree
        // m_useCrossTerms
        // m_exercisableProportion
        m_conditionalExpectation.resize(m_nPaths, 0.0);
        m_premiumBefore.resize(m_nPaths, 0.0);
        m_premiumAfter.resize(m_nPaths, 0.0);
        m_exerciseDecision.resize(m_nPaths, 0.0);
        m_weights.resize(m_nPaths, 1.0);
        const Time nullDate = 0;
        m_exerciseFlows.resize(m_nExercises, AMCFlow(m_nPaths, 0, nullDate, nullDate));
        //for (size_t i = 0; i < m_nExercises; ++i) {
        //    m_exerciseFlows[i] = AMCFlow(m_nPaths, i, exObsDate[i], exSettleDate[i]);
        //}
        // m_performances
    }
    void computeForward() {
        // Don't forget to set the exercise index of the contract flows.
        // Exercise Index of a flow = i <=> obs(i-1) < obs(flow) <= obs(i)
        for (auto & flow : m_exerciseFlows) {
            if (flow.getObservationDate() >= m_modelDate) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    flow.setAmount(j, 1.0);
                }
            }
        }
        for (auto& flow : m_contractFlows) {
            if (flow.getObservationDate() >= m_modelDate) {
                for (size_t j = 0; j < m_nPaths; ++j) {
                    flow.setAmount(j, 1.0);
                }
            }
        }
        m_contractFlows;
        m_basis;
        m_basisWeighted;
        m_stateVariables;
        m_linearStateVariables;
    }
    void computeBackward() {
        // The backward loop.
        size_t exIdx = m_nExercises - 1;
        while (exIdx < m_nExercises && m_exerciseFlows[exIdx].getObservationDate() >= m_modelDate) {
            // This takes the premium vector from the next exercise premium (= m_premiumAfter), discount it to the current exercise observation date,
            // and adds the period flows (discounted to the same date) to the premium vector.
            updatePremiumBefore(exIdx);
            // Rescale state variables.
            rescaleStateVariable(m_stateVariables[exIdx]);
            rescaleStateVariable(m_linearStateVariables[exIdx]);
            // Compute the basis.
            computeBasis(m_stateVariables[exIdx], m_linearStateVariables[exIdx]);
            // Compute the conditional expectation.
            computeConditionalExpectation(exIdx);
            // Compute the exercise decision
            m_exercises[exIdx]->computeExercise(m_exerciseDecision, m_conditionalExpectation, m_performances[exIdx]);
            // Update premiumAfter = exDecision * m_exerciseFlows (= rebate premium) + (1 - exDecision) * premiumBefore (= continuation premium)  
            updatePremiumAfter(exIdx);
            // Update the new flows : 
            // The current exercise is geared by exDecision, the future exercises are geared by (1 - exDecision) = the alive proportion.
            // The future payoff flows are geared by (1 - exDecision), the current payoff flow is not geared (as the flow is included in the rebate)
            updateFutureFlows(exIdx);
            // Decrement the exercise index.
            exIdx--;
        }
        // We end up with the flows being in m_contractFlows and m_exerciseFlows
        // (the exercise and payoff flows strictly before the model date are set to 0 in computeForward())
    }
};

int main()
{
    return 0;
}