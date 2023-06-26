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
    const Time modelDate,
    const Time exerciseDate) :
    m_nPaths(nPaths),
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
    m_individualSmoothings = Matrix(m_nPaths, m_nUnderlyings);
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

void AMCSmoothing_Parameters::getIndividualSmoothing(std::vector<double> const& regressedGain, Matrix const& individualPerformances) const {
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
                indivSmoothingRow[j] = callSpread(barrierShift + (performance / epsilon));
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
void AMCSmoothing_Parameters_Mono::getPerformance(Matrix const& individualPerformances, std::vector<double>& performance) const {
    for (size_t i = 0; i < m_nPaths; ++i) {
        const auto* indivPerfRow = individualPerformances[i];
        assert(indivPerfRow);
        performance[i] = indivPerfRow[0];
    }
}

void AMCSmoothing_Parameters_WorstOf::getPerformance(Matrix const& individualPerformances, std::vector<double>& performance) const {
    for (size_t i = 0; i < m_nPaths; ++i) {
        const auto* indivPerfRow = individualPerformances[i];
        assert(indivPerfRow);
        performance[i] = DBL_MAX;
        for (size_t j = 0; j < m_nUnderlyings; ++j) {
            performance[i] = std::min(performance[i], indivPerfRow[j]);
        }
    }
}

void AMCSmoothing_Parameters_BestOf::getPerformance(Matrix const& individualPerformances, std::vector<double>& performance) const {
    for (size_t i = 0; i < m_nPaths; ++i) {
        const auto* indivPerfRow = individualPerformances[i];
        assert(indivPerfRow);
        performance[i] = -DBL_MAX;
        for (size_t j = 0; j < m_nUnderlyings; ++j) {
            performance[i] = std::max(performance[i], indivPerfRow[j]);
        }
    }
}

/* Computes the smoothing indicator */
void AMCSmoothing_Parameters_Mono::getSmoothing(std::vector<double> const& regressedGain, Matrix const& individualPerformances, std::vector<double>& smoothing) const {
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

void AMCSmoothing_Parameters_Multi::getSmoothing(std::vector<double> const& regressedGain, Matrix const& individualPerformances, std::vector<double>& smoothing) const {
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
