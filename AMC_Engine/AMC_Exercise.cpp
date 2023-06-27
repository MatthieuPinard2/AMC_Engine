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
bool AMCExercise_Putable::isPutable() const {
    return true;
}

void AMCExercise_Putable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const&) const
{
    const size_t nPaths = exercise.size();
    for (size_t i = 0; i < nPaths; ++i) {
        exercise[i] = regressedGain[i] < 0.0 ? 1.0 : 0.0;
    }
}

/* Callable Exercise */
bool AMCExercise_Callable::isCallable() const {
    return true;
}

void AMCExercise_Callable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const&) const
{
    const size_t nPaths = exercise.size();
    for (size_t i = 0; i < nPaths; ++i) {
        exercise[i] = regressedGain[i] > 0.0 ? 1.0 : 0.0;
    }
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
