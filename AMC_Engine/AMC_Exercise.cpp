#include "AMC_Exercise.h"

bool AMCExercise::isCallable() const {
    return false;
}

bool AMCExercise::isPutable() const {
    return false;
}

void AMCExercise::computeWeights(
    std::vector<double>& weights,
    Matrix const&) const
{
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = 1.0;
    }
}

/* Autocallable Exercise */
AMCExercise_Autocallable::AMCExercise_Autocallable(AMCSmoothing_Parameters const& smoothingParams) :
    m_smoothingParams(&smoothingParams) {}

inline void AMCExercise_Autocallable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix const& individualPerformances) const
{
    for (size_t i = 0; i < exercise.size(); ++i) {
        exercise[i] = m_smoothingParams->getSmoothing(regressedGain[i], individualPerformances[i]);
    }
}

void AMCExercise_Autocallable::computeWeights(
    std::vector<double>& weights,
    Matrix const& individualPerformances) const
{
    const size_t nPaths = weights.size();
    double meanPerf = 0.0, stdPerf = 0.0;
    for (size_t i = 0; i < nPaths; ++i) {
        weights[i] = m_smoothingParams->getPerformance(individualPerformances[i]);
    }
    standardDeviation(weights, meanPerf, stdPerf);
    const double bandwidth = 1.06 * stdPerf * pow(double(nPaths), -0.2);
    for (size_t i = 0; i < nPaths; ++i) {
        weights[i] = (weights[i] / bandwidth);
        weights[i] = exp(-0.25 * weights[i] * weights[i]);
    }
}

/* Putable Exercise */
bool AMCExercise_Putable::isPutable() const {
    return true;
}

inline void AMCExercise_Putable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix const&) const
{
    for (size_t i = 0; i < exercise.size(); ++i) {
        exercise[i] = regressedGain[i] <= 0.0 ? 1.0 : 0.0;
    }
}

/* Callable Exercise */
bool AMCExercise_Callable::isCallable() const {
    return true;
}

inline void AMCExercise_Callable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix const&) const
{
    for (size_t i = 0; i < exercise.size(); ++i) {
        exercise[i] = regressedGain[i] >= 0.0 ? 1.0 : 0.0;
    }
}

/* Null Exercise */
inline void AMCExercise_NoExercise::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const&,
    Matrix const&) const
{
    for (size_t i = 0; i < exercise.size(); ++i) {
        exercise[i] = 0.0;
    }
}
