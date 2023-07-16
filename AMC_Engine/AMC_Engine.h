#include "AMC_Math.h"
#include "AMC_Flow.h"
#include "AMC_Exercise.h"

class AMCEngine {
private:
    // Contract Flows (= flows paid when the contract is alive)
    std::vector<AMCFlow> m_contractFlows;
    // Rebate Flows (= flows paid when exercising, i.e. the rebates)
    std::vector<AMCFlow> m_exerciseFlows;
    // Definition of Exercise Events.
    std::vector<AMCExercisePtr> m_exercises;
    size_t m_nExercises;
    // Percentage of the notional that can be exercised. 
    double m_exercisableProportion;
    // AMC Specific Engine Data
    size_t m_polynomialDegree;
    bool m_useCrossTerms;
    // MC Engine Data
    Time m_modelDate;
    size_t m_nPaths;
    // AMC Specific Indicators
    std::vector<double> m_exitProbability;
    std::vector<double> m_exitImpact;
    // Internal data for the Exercise computation.
    std::vector<double> m_conditionalExpectation;
    std::vector<double> m_premiumBefore;
    std::vector<double> m_premiumAfter;
    std::vector<double> m_optimizedPremium;
    std::vector<double> m_exerciseDecision;
    std::vector<double> m_weights;
    Matrix<double> m_basis;
    Matrix<double> m_basisWeighted;
    std::vector<Matrix<double>> m_stateVariables;
    std::vector<Matrix<double>> m_linearStateVariables;
    std::vector<Matrix<double>> m_performances;
    Matrix<size_t> m_monomialExponents;
    size_t m_nStateVariables;
    size_t m_nLinearStateVariables;
    Matrix<double> m_monomialsPerStateVariable;
    bool m_optimizeExercise;
    // Smoothing related.
    double m_smoothingWidth;
    double m_smoothingGearing;
    // Private functions
    void rescaleStateVariable(Matrix<double>& svMatrix) const;
    size_t getBasisSize(const bool withLinearSV) const;
    void computeBasis(Matrix<double>& svMatrix, Matrix<double>& linearSVMatrix);
    void updateFutureFlows(const size_t exIdx);
    void rescaleByWeights(const size_t exIdx);
    bool needRegression(const size_t exIdx);
    void clampConditionalExpectation(const double minimumGain, const double maximumGain);
    void computeConditionalExpectation(const size_t exIdx);
    void computePremiumBeforeExercise(const size_t exIdx);
    void computePremiumAfterExercise(const size_t exIdx);
    void computeIndicators(const size_t exIdx);
    void setFlowExerciseIndex(AMCFlow& flow) const;
    void precomputeMonomialExponents();
    double getSmoothingWidth(const size_t exIdx) const;
public:
    AMCEngine(AMCEngine const&) = delete;
    AMCEngine(AMCEngine&&) = delete;
    AMCEngine& operator=(AMCEngine const&) = delete;
    AMCEngine& operator=(AMCEngine&&) = delete;
    AMCEngine();
    void computeForward();
    void computeBackward();
};
