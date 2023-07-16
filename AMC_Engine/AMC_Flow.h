#include <vector>

using Time = int;
constexpr Time nullDate = Time{ 0 };

struct FlowAmount {
    // Total Amount (Cash + Physical)
    double m_amountTotal;
    // Physical Amount
    double m_amountPhysical;
    // Number of Shares (= Physical / Spot)
    double m_amountShares;
};
constexpr FlowAmount zeroFlow = { .m_amountTotal = 0.0, .m_amountPhysical = 0.0, .m_amountShares = 0.0 };

class AMCFlow {
private:
    size_t m_exerciseIndex;
    Time m_observationDate;
    Time m_settlementDate;
    std::vector<FlowAmount> m_amount;
    // Discount factor from today to the settlement date
    std::vector<double> m_dfToSettle;
    // Discount factor from today to the observation date
    std::vector<double> m_dfToObs;
    // Discount factor from the observation date to the settlement date
    std::vector<double> m_dfObsToSettle;
    // Decides whether the flow is paid if the exercise event is happening.
    bool m_isIncludedInRebate;
    // Such flows are paid regardless of an exercise event happening.
    bool m_isBulletFlow;
public:
    bool isIncludedInRebate() const;
    bool isBulletFlow() const;
    size_t getExerciseIndex() const;
    Time getObservationDate() const;
    Time getSettlementDate() const;
    double getDFToSettlementDate(const size_t i) const;
    double getDFToObservationDate(const size_t i) const;
    double getDFObsToSettleDate(const size_t i) const;
    double getAmount(const size_t i) const;
    void setAmount(const size_t i, const double amountTotal, const double amountPhysical, const double amountShares);
    void scaleAmount(const size_t i, const double scale);
    void setDiscountFactors(const size_t i, const double dfToSettle, const double dfObsToSettle);
    void setExerciseIndex(const size_t exerciseIndex);
    AMCFlow() noexcept;
    AMCFlow(
        const size_t nPaths,
        const size_t exerciseIndex,
        const Time observationDate,
        const Time settlementDate,
        const bool isIncludedInRebate,
        const bool isBulletFlow);
};
