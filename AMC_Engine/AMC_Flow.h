#pragma once
#include <vector>

using Time = int;

class AMCFlow {
private:
    size_t m_exerciseIndex;
    Time m_observationDate;
    Time m_settlementDate;
    std::vector<double> m_amount;
    // Discount factor from today to the settlement date
    std::vector<double> m_dfToSettle;
    // Discount factor from today to the observation date
    std::vector<double> m_dfToObs;
    // Discount factor from the observation date to the settlement date
    std::vector<double> m_dfObsToSettle; 
public:
    inline size_t getExerciseIndex() const;
    inline Time getObservationDate() const;
    inline Time getSettlementDate() const;
    inline double getDFToSettlementDate(const size_t i) const;
    inline double getDFToObservationDate(const size_t i) const;
    inline double getDFObsToSettleDate(const size_t i) const;
    inline double getAmount(const size_t i) const;
    inline void setAmount(const size_t i, const double amount);
    inline double scaleAmount(const size_t i, const double scale);
    inline void setDiscountFactors(const size_t i, const double dfToSettle, const double dfObsToSettle);
    AMCFlow(const size_t nPaths, const size_t exerciseIndex, const Time observationDate, const Time settlementDate);
};
