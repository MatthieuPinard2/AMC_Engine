#include "AMC_Exercise.h"

// Utility for Conditional Callable + Putable.
namespace {
    void computeExerciseForCallPut(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        const double smoothingWidthFactor,
        const bool isCallable)
    {
        const size_t nPaths = exercise.size();
        double mean = 0.0, stdGain = 0.0;
        bool smoothExercise = false;
        const double gainGearing = isCallable ? 1.0 : -1.0;
        // Smoothing of the issuer/holder exercise.
        if (smoothingWidthFactor > 0.0) {
            standardDeviation(regressedGain, mean, stdGain);
            smoothExercise = (stdGain > 0.0);
        }
        if (!smoothExercise) {
            for (size_t i = 0; i < nPaths; ++i) {
                exercise[i] = (gainGearing * regressedGain[i]) > 0.0 ? 1.0 : 0.0;
            }
        }
        else {
            const double smoothingWidth = smoothingWidthFactor * stdGain;
            for (size_t i = 0; i < nPaths; ++i) {
                exercise[i] = AMCSmoothing_Parameters::callSpread(0.5 + (gainGearing * regressedGain[i] / smoothingWidth));
            }
        }
    }

    void computeWeightsForConditional(
        AMCSmoothing_Parameters const& smoothingParams,
        std::vector<double>& weights,
        Matrix<double> const& individualPerformances,
        const bool isAutocallable)
    {
        const size_t nPaths = weights.size();
        double meanPerf = 0.0, stdPerf = 0.0;
        smoothingParams.getPerformance(individualPerformances, weights);
        standardDeviation(weights, meanPerf, stdPerf);
        const double bandwidth = 1.06 * stdPerf * pow(static_cast<double>(nPaths), -0.2);
        // Means the underlying performance has 0 variance.
        if (bandwidth <= 0.0) {
            for (size_t i = 0; i < nPaths; ++i) {
                weights[i] = 1.0;
            }
        }
        // For autocallables, we are only interested at the gain around the barrier, we use a 
        // gaussian kernel centered around the barrier.
        else if (isAutocallable) {
            for (size_t i = 0; i < nPaths; ++i) {
                weights[i] /= bandwidth;
                weights[i] = exp(-0.25 * weights[i] * weights[i]);
            }
        }
        // For conditional Callable/Putable, we are interested at the gain whenever we can exercise
        // (i.e. positive barrier performance), but we keep that gaussian kernel around the barrier.
        else {
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

    void computeExerciseForConditional(
        AMCSmoothing_Parameters const& smoothingParams,
        Matrix<double> const& individualPerformances,
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        const double smoothingWidthFactor,
        const bool isCallable)
    {
        const size_t nPaths = exercise.size();
        std::vector<double> gap(nPaths);
        double mean = 0.0, stdGain = 0.0;
        bool smoothExercise = false;
        const double gainGearing = isCallable ? 1.0 : -1.0;
        // Smoothing of the conditional part.
        // For the gap computation, we are interested in the trajectories having 
        // a positive gain (for Callable) or negative gain (for Putable).
        if (isCallable) {
            for (size_t i = 0; i < nPaths; ++i) {
                gap[i] = std::max(regressedGain[i], DBL_EPSILON);
            }
        }
        else {
            for (size_t i = 0; i < nPaths; ++i) {
                gap[i] = std::min(regressedGain[i], -DBL_EPSILON);
            }
        }
        smoothingParams.getSmoothing(gap, individualPerformances, exercise);
        // Smoothing of the issuer/holder exercise.
        // We are discarding paths where the condition is not fulfilled for the smoothing width computation.
        if (smoothingWidthFactor > 0.0) {
            standardDeviation(regressedGain, exercise, mean, stdGain);
            if (stdGain > 0.0) [[likely]] {
                smoothExercise = true;
            }
        }
        if (!smoothExercise) {
            for (size_t i = 0; i < nPaths; ++i) {
                exercise[i] *= (gainGearing * regressedGain[i]) > 0.0 ? 1.0 : 0.0;
            }
        }
        else {
            const double smoothingWidth = smoothingWidthFactor * stdGain;
            for (size_t i = 0; i < nPaths; ++i) {
                exercise[i] *= AMCSmoothing_Parameters::callSpread(0.5 + (gainGearing * regressedGain[i] / smoothingWidth));
            }
        }
    }
}

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
    computeWeightsForConditional(*m_smoothingParams, weights, individualPerformances, true);
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
    computeExerciseForCallPut(exercise, regressedGain, m_smoothingWidth, isCallable());
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
    computeExerciseForCallPut(exercise, regressedGain, m_smoothingWidth, isCallable());
}

/* Conditional Putable Exercise */
AMCExercise_ConditionalPutable::AMCExercise_ConditionalPutable(
    AMCSmoothing_ParametersConstPtr smoothingParams,
    const double smoothingWidth) :
    m_smoothingParams(smoothingParams),
    m_smoothingWidth(smoothingWidth) {}

bool AMCExercise_ConditionalPutable::isPutable() const {
    return true;
}

void AMCExercise_ConditionalPutable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const& individualPerformances) const
{
    computeExerciseForConditional(*m_smoothingParams, individualPerformances, exercise, regressedGain, m_smoothingWidth, isCallable());
}

void AMCExercise_ConditionalPutable::computeWeights(
    std::vector<double>& weights,
    Matrix<double> const& individualPerformances) const
{
    computeWeightsForConditional(*m_smoothingParams, weights, individualPerformances, false);
}

/* Conditional Callable Exercise */
AMCExercise_ConditionalCallable::AMCExercise_ConditionalCallable(
    AMCSmoothing_ParametersConstPtr smoothingParams,
    const double smoothingWidth) :
    m_smoothingParams(smoothingParams),
    m_smoothingWidth(smoothingWidth) {}

bool AMCExercise_ConditionalCallable::isCallable() const {
    return true;
}

void AMCExercise_ConditionalCallable::computeExercise(
    std::vector<double>& exercise,
    std::vector<double> const& regressedGain,
    Matrix<double> const& individualPerformances) const
{
    computeExerciseForConditional(*m_smoothingParams, individualPerformances, exercise, regressedGain, m_smoothingWidth, isCallable());
}

void AMCExercise_ConditionalCallable::computeWeights(
    std::vector<double>& weights,
    Matrix<double> const& individualPerformances) const
{
    computeWeightsForConditional(*m_smoothingParams, weights, individualPerformances, false);
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
