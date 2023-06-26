#include "AMC_Smoothing.h"

AMCSmoothing_Parameters::AMCSmoothing_Parameters(
    std::vector<double> const& deltaMax,
    std::vector<double> const& spreadMin,
    std::vector<double> const& spreadMax,
    std::vector<double> const& barrierLevel,
    std::vector<double> const& FX,
    const double notional,
    const double smoothingGearing,
    const BarrierType barrierType,
    const Time modelDate,
    const Time exerciseDate) :
    m_spreadMin(spreadMin),
    m_spreadMax(spreadMax),
    m_deltaMax(deltaMax),
    m_barrierLevel(barrierLevel),
    m_FX(FX),
    m_notional(notional),
    m_smoothingGearing(smoothingGearing),
    m_barrierType(barrierType),
    m_disableSmoothing(exerciseDate <= modelDate) {
    m_nUnderlyings = deltaMax.size();
    m_adjustedDMax.resize(m_nUnderlyings);
    m_perfGearing = (m_barrierType == BarrierType::UpBarrier) ? 1.0 : -1.0;
    m_individualSmoothings.resize(m_nUnderlyings);
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
double AMCSmoothing_Parameters::callSpread(const double x) const {
    const double y = std::max(0.0, std::min(x, 1.0));
    return (y <= 0.5) ? 2.0 * y * y : (4.0 - 2.0 * y) * y - 1.0;
}

double AMCSmoothing_Parameters::callSpreadUnsmoothed(const double x) const {
    return x >= 0.0 ? 1.0 : 0.0;
}

double AMCSmoothing_Parameters::getIndividualSmoothing(const double regressedGain, const double* individualPerformances, const size_t i) const {
    assert(individualPerformances);
    const double performance = m_perfGearing * (individualPerformances[i] - m_barrierLevel[i]);
    if (m_disableSmoothing || m_smoothingGearing <= 0.0) [[unlikely]] {
        return callSpreadUnsmoothed(performance);
    }
    else {
        const double premiumGap = regressedGain * m_notional;
        double epsilon = std::abs(premiumGap / m_adjustedDMax[i]);
        epsilon = std::max(m_spreadMin[i], std::min(epsilon, m_spreadMax[i])) * m_smoothingGearing;
        const double barrierShift = (premiumGap >= 0.0) ? 1.0 : 0.0;
        return callSpread(barrierShift + (performance / epsilon));
    }
}

/* Computing the performance given the individual individualPerformances */
double AMCSmoothing_Parameters_Mono::getPerformance(const double* individualPerformances) const {
    assert(individualPerformances); 
    return individualPerformances[0];
}

double AMCSmoothing_Parameters_WorstOf::getPerformance(const double* individualPerformances) const {
    assert(individualPerformances); 
    double performance = DBL_MAX;
    for (size_t i = 0; i < m_nUnderlyings; ++i) {
        performance = std::min(performance, individualPerformances[i]);
    }
    return performance;
}

double AMCSmoothing_Parameters_BestOf::getPerformance(const double* individualPerformances) const {
    assert(individualPerformances);
    double performance = -DBL_MAX;
    for (size_t i = 0; i < m_nUnderlyings; ++i) {
        performance = std::max(performance, individualPerformances[i]);
    }
    return performance;
}

/* Computes the smoothing indicator */
double AMCSmoothing_Parameters_Mono::getSmoothing(const double regressedGain, const double* individualPerformances) const {
    assert(individualPerformances); 
    return getIndividualSmoothing(regressedGain, individualPerformances, 0);
}

bool AMCSmoothing_Parameters_WorstOf::takeMinimumOfSmoothings() const {
    return m_barrierType == BarrierType::UpBarrier;
}

bool AMCSmoothing_Parameters_BestOf::takeMinimumOfSmoothings() const {
    return m_barrierType == BarrierType::DownBarrier;
}

double AMCSmoothing_Parameters_Multi::getSmoothing(const double regressedGain, const double* individualPerformances) const {
    for (size_t i = 0; i < m_nUnderlyings; ++i) {
        m_individualSmoothings[i] = getIndividualSmoothing(regressedGain, individualPerformances, i);
    }
    double smoothing = 1.0;
    if (takeMinimumOfSmoothings()) [[likely]] {
        for (size_t i = 0; i < m_nUnderlyings; ++i) {
            smoothing *= m_individualSmoothings[i];
        }
        return smoothing;
    }
    else {
        for (size_t i = 0; i < m_nUnderlyings; ++i) {
            smoothing *= 1.0 - m_individualSmoothings[i];
        }
        return 1.0 - smoothing;
    }
}
