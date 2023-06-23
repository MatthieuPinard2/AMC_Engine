#pragma once
#include "AMC_Smoothing.h"
#include "AMC_Math.h"

class AMCExercise {
public:
    virtual ~AMCExercise() = default;
    virtual bool isCallable() const;
    virtual bool isPutable() const;
    virtual void computeWeights(
        std::vector<double>& weights,
        Matrix const& individualPerformances) const;
    virtual inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix const& individualPerformances) const = 0;
};

typedef std::shared_ptr<AMCExercise> AMCExercisePtr;

/* Autocallable Exercise */
class AMCExercise_Autocallable : protected AMCExercise {
    AMCSmoothing_ParametersConstPtr m_smoothingParams;
public:
    AMCExercise_Autocallable(AMCSmoothing_Parameters const& smoothingParams);
    inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix const& individualPerformances) const;
    virtual void computeWeights(
        std::vector<double>& weights,
        Matrix const& individualPerformances) const;
};

/* Putable Exercise */
class AMCExercise_Putable : protected AMCExercise {
public:
    virtual bool isPutable() const;
    inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix const& individualPerformances) const;
};

/* Callable Exercise */
class AMCExercise_Callable : protected AMCExercise {
public:
    virtual bool isCallable() const;
    inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix const& individualPerformances) const;
};

/* Null Exercise */
class AMCExercise_NoExercise : protected AMCExercise {
public:
    inline void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix const& individualPerformances) const;
};