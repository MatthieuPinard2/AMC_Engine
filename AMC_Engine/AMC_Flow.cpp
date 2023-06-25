#include "AMC_Flow.h"

size_t AMCFlow::getExerciseIndex() const {
    return m_exerciseIndex;
}

Time AMCFlow::getObservationDate() const {
    return m_observationDate;
}

Time AMCFlow::getSettlementDate() const {
    return m_settlementDate;
}

double AMCFlow::getDFToSettlementDate(const size_t i) const {
    return m_dfToSettle[i];
}

double AMCFlow::getDFToObservationDate(const size_t i) const {
    return m_dfToObs[i];
}

double AMCFlow::getDFObsToSettleDate(const size_t i) const {
    return m_dfObsToSettle[i];
}

double AMCFlow::getAmount(const size_t i) const {
    return m_amount[i];
}

void AMCFlow::setAmount(const size_t i, const double amount) {
    m_amount[i] = amount;
}

void AMCFlow::scaleAmount(const size_t i, const double scale) {
    m_amount[i] *= scale;
}

void AMCFlow::setDiscountFactors(const size_t i, const double dfToSettle, const double dfObsToSettle) {
    m_dfToSettle[i] = dfToSettle;
    m_dfToObs[i] = dfToSettle / dfObsToSettle;
    m_dfObsToSettle[i] = dfObsToSettle;
}

bool AMCFlow::isIncludedInRebate() const {
    return m_isIncludedInRebate;
}

bool AMCFlow::isBulletFlow() const {
    return m_isBulletFlow;
}

void AMCFlow::setExerciseIndex(const size_t exerciseIndex) {
    m_exerciseIndex = exerciseIndex;
}

AMCFlow::AMCFlow() :
    m_exerciseIndex(0),
    m_observationDate(nullDate),
    m_settlementDate(nullDate),
    m_isIncludedInRebate(true),
    m_isBulletFlow(false) {
}

AMCFlow::AMCFlow(
    const size_t nPaths,
    const size_t exerciseIndex,
    const Time observationDate,
    const Time settlementDate,
    const bool isIncludedInRebate,
    const bool isBulletFlow) :
    m_exerciseIndex(exerciseIndex),
    m_observationDate(observationDate),
    m_settlementDate(settlementDate),
    m_isIncludedInRebate(isIncludedInRebate),
    m_isBulletFlow(isBulletFlow) {
    m_amount.resize(nPaths, 0.0);
    m_dfToSettle.resize(nPaths, 0.0);
    m_dfToObs.resize(nPaths, 0.0);
    m_dfObsToSettle.resize(nPaths, 0.0);
}
