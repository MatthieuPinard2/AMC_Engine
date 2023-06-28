#include "AMC_Exercise.h"

bool AMCExercise::isCallable() const {
    return false;
}

bool AMCExercise::isPutable() const {
    return false;
}

bool AMCExercise::isNoExercise() const {
    return false;
}

void AMCExercise::computeWeights(
    std::vector<double>& weights,
    Matrix<double> const&) const
{
    const size_t nPaths = weights.size();
    for (size_t i = 0; i < nPaths; ++i) {
        weights[i] = 1.0;
    }
}

/* Autocallable Exercise */
AMCExercise_Autocallable::AMCExercise_Autocallable(AMCSmoothing_ParametersConstPtr smoothingParams) :
    m_smoothingParams(smoothingParams) {}

void AMCExercise_Autocallable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const& individualPerformances) const
{
    m_smoothingParams->getSmoothing(regressedGain, individualPerformances, exercise);
}

void AMCExercise_Autocallable::computeWeights(
    std::vector<double>& weights,
    Matrix<double> const& individualPerformances) const
{
    const size_t nPaths = weights.size();
    double meanPerf = 0.0, stdPerf = 0.0;
    m_smoothingParams->getPerformance(individualPerformances, weights);
    standardDeviation(weights, meanPerf, stdPerf);
    const double bandwidth = 1.06 * stdPerf * pow(static_cast<double>(nPaths), -0.2);
    for (size_t i = 0; i < nPaths; ++i) {
        weights[i] = (weights[i] / bandwidth);
        weights[i] = exp(-0.25 * weights[i] * weights[i]);
    }
}

/* Putable Exercise */
AMCExercise_Putable::AMCExercise_Putable(const double smoothingWidth) noexcept
    : m_smoothingWidth(smoothingWidth) {}

bool AMCExercise_Putable::isPutable() const {
    return true;
}

void AMCExercise_Putable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const&) const
{
    const size_t nPaths = exercise.size();
    double mean = 0.0, stdGain = 0.0;
    bool smoothExercise = false;
    if (m_smoothingWidth > 0.0) {
        standardDeviation(regressedGain, mean, stdGain);
        smoothExercise = (stdGain > 0.0);
    }
    if (!smoothExercise) {
        for (size_t i = 0; i < nPaths; ++i) {
            exercise[i] = regressedGain[i] < 0.0 ? 1.0 : 0.0;
        }
    }
    else {
        const double smoothingWidth = m_smoothingWidth * stdGain;
        for (size_t i = 0; i < nPaths; ++i) {
            exercise[i] = AMCSmoothing_Parameters::callSpread((0.5 * smoothingWidth - regressedGain[i]) / smoothingWidth);
        }
    }
}

/* Callable Exercise */
AMCExercise_Callable::AMCExercise_Callable(const double smoothingWidth) noexcept
    : m_smoothingWidth(smoothingWidth) {}

bool AMCExercise_Callable::isCallable() const {
    return true;
}

void AMCExercise_Callable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const&) const
{
    const size_t nPaths = exercise.size();
    double mean = 0.0, stdGain = 0.0;
    bool smoothExercise = false;
    if (m_smoothingWidth > 0.0) {
        standardDeviation(regressedGain, mean, stdGain);
        smoothExercise = (stdGain > 0.0);
    }
    if (!smoothExercise) {
        for (size_t i = 0; i < nPaths; ++i) {
            exercise[i] = regressedGain[i] > 0.0 ? 1.0 : 0.0;
        }
    }
    else {
        const double smoothingWidth = m_smoothingWidth * stdGain;
        for (size_t i = 0; i < nPaths; ++i) {
            exercise[i] = AMCSmoothing_Parameters::callSpread((0.5 * smoothingWidth + regressedGain[i]) / smoothingWidth);
        }
    }
}

// Utility for Conditional Callable + Putable.
namespace {
    void computeWeightsForConditional(
        AMCSmoothing_Parameters const& smoothingParams,
        std::vector<double>& weights,
        Matrix<double> const& individualPerformances)
    {
        const size_t nPaths = weights.size();
        double meanPerf = 0.0, stdPerf = 0.0;
        smoothingParams.getPerformance(individualPerformances, weights);
        standardDeviation(weights, meanPerf, stdPerf);
        const double bandwidth = 1.06 * stdPerf * pow(static_cast<double>(nPaths), -0.2);
        for (size_t i = 0; i < nPaths; ++i) {
            if (weights[i] >= 0.0) {
                weights[i] = 1.0;
            }
            else {
                weights[i] /= bandwidth;
                weights[i] = exp(-0.25 * weights[i] * weights[i]);
            }
        }
    }
}

/* Conditional Putable Exercise */
AMCExercise_ConditionalPutable::AMCExercise_ConditionalPutable(AMCSmoothing_ParametersConstPtr smoothingParams) :
    m_smoothingParams(smoothingParams) {}

bool AMCExercise_ConditionalPutable::isPutable() const {
    return true;
}

void AMCExercise_ConditionalPutable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const& individualPerformances) const
{
    const size_t nPaths = exercise.size();
    std::vector<double> gap(nPaths);
    for (size_t i = 0; i < nPaths; ++i) {
        gap[i] = std::min(regressedGain[i], -DBL_EPSILON);
    }
    m_smoothingParams->getSmoothing(gap, individualPerformances, exercise);
    for (size_t i = 0; i < nPaths; ++i) {
        exercise[i] *= regressedGain[i] < 0.0 ? 1.0 : 0.0;
    }
}

void AMCExercise_ConditionalPutable::computeWeights(
    std::vector<double>& weights,
    Matrix<double> const& individualPerformances) const
{
    computeWeightsForConditional(*m_smoothingParams, weights, individualPerformances);
}

/* Conditional Callable Exercise */
AMCExercise_ConditionalCallable::AMCExercise_ConditionalCallable(AMCSmoothing_ParametersConstPtr smoothingParams) :
    m_smoothingParams(smoothingParams) {}

bool AMCExercise_ConditionalCallable::isCallable() const {
    return true;
}

void AMCExercise_ConditionalCallable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const& individualPerformances) const
{
    const size_t nPaths = exercise.size();
    std::vector<double> gap(nPaths);
    for (size_t i = 0; i < nPaths; ++i) {
        gap[i] = std::max(regressedGain[i], DBL_EPSILON);
    }
    m_smoothingParams->getSmoothing(gap, individualPerformances, exercise);
    for (size_t i = 0; i < nPaths; ++i) {
        exercise[i] *= regressedGain[i] > 0.0 ? 1.0 : 0.0;
    }
}

void AMCExercise_ConditionalCallable::computeWeights(
    std::vector<double>& weights,
    Matrix<double> const& individualPerformances) const
{
    computeWeightsForConditional(*m_smoothingParams, weights, individualPerformances);
}

/* Null Exercise */
void AMCExercise_NoExercise::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const&,
    Matrix<double> const&) const
{
    const size_t nPaths = exercise.size();
    for (size_t i = 0; i < nPaths; ++i) {
        exercise[i] = 0.0;
    }
}

bool AMCExercise_NoExercise::isNoExercise() const {
    return true;
}
