#include "AMC_Smoothing.h"
#include "AMC_Math.h"

class AMCExercise {
public:
    virtual ~AMCExercise() = default;
    virtual bool isCallable() const;
    virtual bool isPutable() const;
    virtual bool isNoExercise() const;
    virtual void computeWeights(
        std::vector<double>& weights,
        Matrix<double> const& individualPerformances) const;
    virtual void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix<double> const& individualPerformances) const = 0;
};

typedef std::shared_ptr<AMCExercise> AMCExercisePtr;

/* Autocallable Exercise */
class AMCExercise_Autocallable : public AMCExercise {
private:
    AMCSmoothing_ParametersConstPtr m_smoothingParams;
public:
    AMCExercise_Autocallable(AMCSmoothing_ParametersConstPtr smoothingParams);
    void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix<double> const& individualPerformances) const override;
    void computeWeights(
        std::vector<double>& weights,
        Matrix<double> const& individualPerformances) const override;
};

/* Putable Exercise */
class AMCExercise_Putable : public AMCExercise {
private:
    double m_smoothingWidth;
public:
    AMCExercise_Putable(const double smoothingWidth = 0.0) noexcept;
    bool isPutable() const override;
    void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix<double> const& individualPerformances) const override;
};

/* Callable Exercise */
class AMCExercise_Callable : public AMCExercise {
private:
    double m_smoothingWidth;
public:
    AMCExercise_Callable(const double smoothingWidth = 0.0) noexcept;
    bool isCallable() const override;
    void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix<double> const& individualPerformances) const override;
};

/* Conditional Putable Exercise */
class AMCExercise_ConditionalPutable : public AMCExercise {
private:
    AMCSmoothing_ParametersConstPtr m_smoothingParams;
    double m_smoothingWidth;
public:
    AMCExercise_ConditionalPutable(
        AMCSmoothing_ParametersConstPtr smoothingParams,
        const double smoothingWidth = 0.0);
    bool isPutable() const override;
    void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix<double> const& individualPerformances) const override;
    void computeWeights(
        std::vector<double>& weights,
        Matrix<double> const& individualPerformances) const override;
};

/* Conditional Callable Exercise */
class AMCExercise_ConditionalCallable : public AMCExercise {
private:
    AMCSmoothing_ParametersConstPtr m_smoothingParams;
    double m_smoothingWidth;
public:
    AMCExercise_ConditionalCallable(
        AMCSmoothing_ParametersConstPtr smoothingParams,
        const double smoothingWidth = 0.0);
    bool isCallable() const override;
    void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix<double> const& individualPerformances) const override;
    void computeWeights(
        std::vector<double>& weights,
        Matrix<double> const& individualPerformances) const override;
};

/* Null Exercise */
class AMCExercise_NoExercise : public AMCExercise {
public:
    bool isNoExercise() const override;
    void computeExercise(
        std::vector<double>& exercise,
        std::vector<double> const& regressedGain,
        Matrix<double> const& individualPerformances) const override;
};
