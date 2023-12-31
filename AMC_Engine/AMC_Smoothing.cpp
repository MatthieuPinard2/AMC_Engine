#include "AMC_Smoothing.h"

AMCSmoothing_Parameters::AMCSmoothing_Parameters(
    const size_t nPaths,
    std::vector<double> const& deltaMax,
    std::vector<double> const& spreadMin,
    std::vector<double> const& spreadMax,
    std::vector<double> const& barrierLevel,
    std::vector<double> const& FX,
    const double notional,
    const double smoothingGearing,
    const BarrierType barrierType,
    const bool disableSmoothing) :
    m_nPaths(nPaths),
    m_spreadMin(spreadMin),
    m_spreadMax(spreadMax),
    m_deltaMax(deltaMax),
    m_barrierLevel(barrierLevel),
    m_FX(FX),
    m_notional(notional),
    m_smoothingGearing(smoothingGearing),
    m_barrierType(barrierType),
    m_disableSmoothing(disableSmoothing) {
    m_nUnderlyings = deltaMax.size();
    m_adjustedDMax.resize(m_nUnderlyings);
    m_perfGearing = (m_barrierType == BarrierType::UpBarrier) ? 1.0 : -1.0;
    m_individualSmoothings = Matrix<double>(m_nPaths, m_nUnderlyings);
    adjustDeltaMax();
}

void AMCSmoothing_Parameters::adjustDeltaMax() {
    for (size_t i = 0; i < m_nUnderlyings; ++i) {
        m_adjustedDMax[i] = m_deltaMax[i] * m_FX[i];
    }
}

size_t AMCSmoothing_Parameters::getUnderlyingsCount() const {
    return m_nUnderlyings;
}

/* Utility functions for the call spread smoothing. */
double AMCSmoothing_Parameters::callSpread(const double x) {
    const double y = std::max(0.0, std::min(x, 1.0));
    return (y <= 0.5) ? (2.0 * y * y) : ((4.0 - 2.0 * y) * y - 1.0);
}

double AMCSmoothing_Parameters::callSpreadUnsmoothed(const double x) {
    return x >= 0.0 ? 1.0 : 0.0;
}

void AMCSmoothing_Parameters::getIndividualSmoothing(std::vector<double> const& regressedGain, Matrix<double> const& individualPerformances) const {
    if (!m_disableSmoothing && m_smoothingGearing > 0.0) [[likely]] {
        for (size_t i = 0; i < m_nPaths; ++i) {
            auto* indivSmoothingRow = m_individualSmoothings[i];
            assert(indivSmoothingRow);
            const auto* indivPerfRow = individualPerformances[i];
            assert(indivPerfRow);
            const double premiumGap = regressedGain[i] * m_notional;
            const double barrierShift = (premiumGap >= 0.0) ? 1.0 : 0.0;
            for (size_t j = 0; j < m_nUnderlyings; ++j) {
                const double performance = m_perfGearing * (indivPerfRow[j] - m_barrierLevel[j]);
                double epsilon = std::abs(premiumGap / m_adjustedDMax[j]);
                epsilon = std::max(m_spreadMin[j], std::min(epsilon, m_spreadMax[j])) * m_smoothingGearing;
                indivSmoothingRow[j] = (epsilon > 0.0) ? callSpread(barrierShift + (performance / epsilon)) : callSpreadUnsmoothed(performance);
            }
        }
    }
    else {
        for (size_t i = 0; i < m_nPaths; ++i) {
            auto* indivSmoothingRow = m_individualSmoothings[i];
            assert(indivSmoothingRow);
            const auto* indivPerfRow = individualPerformances[i];
            assert(indivPerfRow);
            for (size_t j = 0; j < m_nUnderlyings; ++j) {
                const double performance = m_perfGearing * (indivPerfRow[j] - m_barrierLevel[j]);
                indivSmoothingRow[j] = callSpreadUnsmoothed(performance);
            }
        }
    }
}

/* Computing the performance given the individual individualPerformances */
void AMCSmoothing_Parameters_Mono::getPerformance(Matrix<double> const& individualPerformances, std::vector<double>& performance) const {
    for (size_t i = 0; i < m_nPaths; ++i) {
        const auto* indivPerfRow = individualPerformances[i];
        assert(indivPerfRow);
        performance[i] = m_perfGearing * (indivPerfRow[0] - m_barrierLevel[0]);
    }
}

void AMCSmoothing_Parameters_WorstOf::getPerformance(Matrix<double> const& individualPerformances, std::vector<double>& performance) const {
    for (size_t i = 0; i < m_nPaths; ++i) {
        const auto* indivPerfRow = individualPerformances[i];
        assert(indivPerfRow);
        performance[i] = DBL_MAX;
        for (size_t j = 0; j < m_nUnderlyings; ++j) {
            performance[i] = std::min(performance[i], indivPerfRow[j] - m_barrierLevel[j]);
        }
        performance[i] *= m_perfGearing;
    }
}

void AMCSmoothing_Parameters_BestOf::getPerformance(Matrix<double> const& individualPerformances, std::vector<double>& performance) const {
    for (size_t i = 0; i < m_nPaths; ++i) {
        const auto* indivPerfRow = individualPerformances[i];
        assert(indivPerfRow);
        performance[i] = -DBL_MAX;
        for (size_t j = 0; j < m_nUnderlyings; ++j) {
            performance[i] = std::max(performance[i], indivPerfRow[j] - m_barrierLevel[j]);
        }
        performance[i] *= m_perfGearing;
    }
}

/* Computes the smoothing indicator */
void AMCSmoothing_Parameters_Mono::getSmoothing(std::vector<double> const& regressedGain, Matrix<double> const& individualPerformances, std::vector<double>& smoothing) const {
    getIndividualSmoothing(regressedGain, individualPerformances);
    for (size_t i = 0; i < m_nPaths; ++i) {
        smoothing[i] = m_individualSmoothings[i][0];
    }
}

bool AMCSmoothing_Parameters_WorstOf::takeMinimumOfSmoothings() const {
    return m_barrierType == BarrierType::UpBarrier;
}

bool AMCSmoothing_Parameters_BestOf::takeMinimumOfSmoothings() const {
    return m_barrierType == BarrierType::DownBarrier;
}

void AMCSmoothing_Parameters_Multi::getSmoothing(std::vector<double> const& regressedGain, Matrix<double> const& individualPerformances, std::vector<double>& smoothing) const {
    getIndividualSmoothing(regressedGain, individualPerformances);
    if (takeMinimumOfSmoothings()) [[likely]] {
        for (size_t i = 0; i < m_nPaths; ++i) {
            double& smooth = smoothing[i];
            const auto* indivSmoothRow = m_individualSmoothings[i];
            assert(indivSmoothRow);
            smooth = 1.0;
            for (size_t j = 0; j < m_nUnderlyings; ++j) {
                smooth *= indivSmoothRow[j];
            }
        }
    }
    else {
        for (size_t i = 0; i < m_nPaths; ++i) {
            double& smooth = smoothing[i];
            const auto* indivSmoothRow = m_individualSmoothings[i];
            assert(indivSmoothRow);
            smooth = 1.0;
            for (size_t j = 0; j < m_nUnderlyings; ++j) {
                smooth *= (1.0 - indivSmoothRow[j]);
            }
            smooth = 1.0 - smooth;
        }
    }
}

// Utility to convert to a shared_ptr
template <class T, class... Args>
auto toSmoothingSharedPtr(Args&&... args) {
    const T smoothingParams(args...);
    return std::make_shared<const T>(std::move(smoothingParams));
}

AMCSmoothing_ParametersConstPtr createSmoothingParameters(
    const UnderlyingType underlyingType,
    const size_t nPaths,
    std::vector<double> const& deltaMax,
    std::vector<double> const& spreadMin,
    std::vector<double> const& spreadMax,
    std::vector<double> const& barrierLevel,
    std::vector<double> const& FX,
    const double notional,
    const double smoothingGearing,
    const BarrierType barrierType,
    const Time modelDate,
    const Time exerciseDate)
{
    const bool disableSmoothing = (exerciseDate <= modelDate);
    if (underlyingType == UnderlyingType::Mono) {
        return toSmoothingSharedPtr<AMCSmoothing_Parameters_Mono>(
            nPaths,
            deltaMax,
            spreadMin,
            spreadMax,
            barrierLevel,
            FX,
            notional,
            smoothingGearing,
            barrierType,
            disableSmoothing
        );
    }
    if (underlyingType == UnderlyingType::WorstOf) {
        return toSmoothingSharedPtr<AMCSmoothing_Parameters_WorstOf>(
            nPaths,
            deltaMax,
            spreadMin,
            spreadMax,
            barrierLevel,
            FX,
            notional,
            smoothingGearing,
            barrierType,
            disableSmoothing
        );
    }
    if (underlyingType == UnderlyingType::BestOf) {
        return toSmoothingSharedPtr<AMCSmoothing_Parameters_BestOf>(
            nPaths,
            deltaMax,
            spreadMin,
            spreadMax,
            barrierLevel,
            FX,
            notional,
            smoothingGearing,
            barrierType,
            disableSmoothing
        );
    }
    return AMCSmoothing_ParametersConstPtr();
}
