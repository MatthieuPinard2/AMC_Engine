#include <vector>
#include <cassert>
#include <memory>
#include "AMC_Math.h"

enum class BarrierType { UpBarrier, DownBarrier };
enum class UnderlyingType { WorstOf, BestOf, Mono };
using Time = int;

class AMCSmoothing_Parameters {
protected:
    size_t m_nPaths;
    size_t m_nUnderlyings;
    std::vector<double> m_spreadMin;     // Floors the minimum smoothing width.
    std::vector<double> m_spreadMax;     // Caps the minimum smoothing width.
    std::vector<double> m_deltaMax;      // Maximum Delta Cash expressed in underlying currency, at today. 
    std::vector<double> m_barrierLevel;  // Barrier *relative* level per udl.
    std::vector<double> m_FX;            // Udl/Payoff FX spot rate. 
    double m_notional;                   // Notional in Payoff currency.
    double m_smoothingGearing;           // Allows to apply gearing on a specific barrier.
    mutable Matrix m_individualSmoothings;
    std::vector<double> m_adjustedDMax;
    double m_perfGearing;
    BarrierType m_barrierType;
    bool m_disableSmoothing;             // If barrierDate <= modelDate, we don't smooth.
    double callSpread(const double x) const;
    double callSpreadUnsmoothed(const double x) const;
    void getIndividualSmoothing(std::vector<double> const& regressedGain, Matrix const& individualPerformances) const;
    void adjustDeltaMax();
public:
    virtual ~AMCSmoothing_Parameters() = default;
    AMCSmoothing_Parameters() = delete;
    AMCSmoothing_Parameters(
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
        const Time exerciseDate);
    size_t getUnderlyingsCount() const;
    virtual void getSmoothing(std::vector<double> const& regressedGain, Matrix const& individualPerformances, std::vector<double>& smoothing) const = 0;
    virtual void getPerformance(Matrix const& individualPerformances, std::vector<double>& performance) const = 0;
};

typedef std::shared_ptr<AMCSmoothing_Parameters> AMCSmoothing_ParametersPtr;
typedef std::shared_ptr<const AMCSmoothing_Parameters> AMCSmoothing_ParametersConstPtr;

class AMCSmoothing_Parameters_Mono : public AMCSmoothing_Parameters {
public:
    void getPerformance(Matrix const& individualPerformances, std::vector<double>& performance) const override;
    void getSmoothing(std::vector<double> const& regressedGain, Matrix const& individualPerformances, std::vector<double>& smoothing) const override;
    AMCSmoothing_Parameters_Mono() = delete;
};

class AMCSmoothing_Parameters_Multi : public AMCSmoothing_Parameters {
private:
    virtual bool takeMinimumOfSmoothings() const = 0;
public:
    void getSmoothing(std::vector<double> const& regressedGain, Matrix const& individualPerformances, std::vector<double>& smoothing) const override;
    AMCSmoothing_Parameters_Multi() = delete;
};

class AMCSmoothing_Parameters_WorstOf : public AMCSmoothing_Parameters_Multi {
private:
    bool takeMinimumOfSmoothings() const override;
public:
    void getPerformance(Matrix const& individualPerformances, std::vector<double>& performance) const override;
    AMCSmoothing_Parameters_WorstOf() = delete;
};

class AMCSmoothing_Parameters_BestOf : public AMCSmoothing_Parameters_Multi {
private:
    bool takeMinimumOfSmoothings() const override;
public:
    void getPerformance(Matrix const& individualPerformances, std::vector<double>& performance) const override;
    AMCSmoothing_Parameters_BestOf() = delete;
};
