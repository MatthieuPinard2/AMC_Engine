#include "AMC_Flow.h"

inline size_t AMCFlow::getExerciseIndex() const {
    return m_exerciseIndex;
}

inline Time AMCFlow::getObservationDate() const {
    return m_observationDate;
}

inline Time AMCFlow::getSettlementDate() const {
    return m_settlementDate;
}

inline double AMCFlow::getDFToSettlementDate(const size_t i) const {
    return m_dfToSettle[i];
}

inline double AMCFlow::getDFToObservationDate(const size_t i) const {
    return m_dfToObs[i];
}

inline double AMCFlow::getDFObsToSettleDate(const size_t i) const {
    return m_dfObsToSettle[i];
}

inline double AMCFlow::getAmount(const size_t i) const {
    return m_amount[i];
}

inline void AMCFlow::setAmount(const size_t i, const double amount) {
    m_amount[i] = amount;
}

inline double AMCFlow::scaleAmount(const size_t i, const double scale) {
    m_amount[i] *= scale;
}

inline void AMCFlow::setDiscountFactors(const size_t i, const double dfToSettle, const double dfObsToSettle) {
    m_dfToSettle[i] = dfToSettle;
    m_dfToObs[i] = dfToSettle / dfObsToSettle;
    m_dfObsToSettle[i] = dfObsToSettle;
}

AMCFlow::AMCFlow(const size_t nPaths, const size_t exerciseIndex, const Time observationDate, const Time settlementDate) :
    m_exerciseIndex(exerciseIndex),
    m_observationDate(observationDate),
    m_settlementDate(settlementDate) {
    m_amount.resize(nPaths, 0.0);
    m_dfToSettle.resize(nPaths, 0.0);
    m_dfToObs.resize(nPaths, 0.0);
    m_dfObsToSettle.resize(nPaths, 0.0);
}
