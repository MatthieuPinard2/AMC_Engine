#include "AMC_Engine.h"
#include <cassert>

double AMCEngine::getSmoothingWidth(const size_t exIdx) const {
    // Disable the smoothing at model date.
    if (m_exerciseFlows[exIdx].getObservationDate() <= m_modelDate)
        return 0.0;
    return m_smoothingWidth * m_smoothingGearing;
}

void AMCEngine::precomputeMonomialExponents() {
    // Do not enter this costly function if we are dealing with a trivial basis.
    if (!m_useCrossTerms || m_nStateVariables <= 1)
        return;
    // The basis would be huge. We don't compute it for performance reasons, and because the 
    // SVD regression would be of very low quality.
    assert(m_nStateVariables <= 4);
    size_t insertIdx = 0;
    if (m_nStateVariables == 2) {
        for (size_t i = 0; i <= m_polynomialDegree; ++i) {
            for (size_t j = 0; j <= i; ++j, ++insertIdx) {
                auto row = m_monomialExponents[insertIdx];
                assert(row);
                row[0] = j; row[1] = i - j;
            }
        }
    }
    else if (m_nStateVariables == 3) {
        for (size_t i = 0; i <= m_polynomialDegree; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                for (size_t k = 0; k <= i - j; ++k, ++insertIdx) {
                    auto row = m_monomialExponents[insertIdx];
                    assert(row);
                    row[0] = j; row[1] = k; row[2] = i - j - k;
                }
            }
        }
    }
    else if (m_nStateVariables == 4) {
        for (size_t i = 0; i <= m_polynomialDegree; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                for (size_t k = 0; k <= i - j; ++k) {
                    for (size_t l = 0; l <= i - j - k; ++l, ++insertIdx) {
                        auto row = m_monomialExponents[insertIdx];
                        assert(row);
                        row[0] = j; row[1] = k; row[2] = l; row[3] = i - j - k - l;
                    }
                }
            }
        }
    }
}

void AMCEngine::rescaleStateVariable(Matrix<double>& svMatrix) const {
    const size_t nStateVariables = svMatrix.getNbCols();
    if (!nStateVariables)
        return;
    std::vector<double> meanSV(nStateVariables), stdSV(nStateVariables);
    standardDeviation(svMatrix, meanSV, stdSV);
    // State Variables with 0 variance will be replaced by ones. (so the system is rank one)
    for (size_t j = 0; j < nStateVariables; ++j) {
        if (stdSV[j] <= 0.0) {
            meanSV[j] -= 1.0;
            stdSV[j] = 1.0;
        }
    }
    // Other State Variables will have mean 0 and variance 1.
    for (size_t i = 0; i < m_nPaths; ++i) {
        auto svRow = svMatrix[i];
        assert(svRow);
        for (size_t j = 0; j < nStateVariables; ++j) {
            svRow[j] -= meanSV[j];
            svRow[j] /= stdSV[j];
        }
    }
}

size_t AMCEngine::getBasisSize(const bool withLinearSV) const {
    size_t basisSize;
    if (!m_useCrossTerms || m_nStateVariables <= 1) {
        // The basis is (1, X_1^1 ... X_1^n, X_m^1 ... X_m^n)
        // This is of size 1 + m * n.
        basisSize = 1 + m_nStateVariables * m_polynomialDegree;
    }
    else {
        // The basis is the set of monomials of m variables, and of degree <= n
        // This is of (m + n) choose n = (m + n)! / (m! * n!) = ((m + 1) * ... * (m + n)) / (1 * ... * n).
        basisSize = 1;
        for (size_t i = m_nStateVariables + 1; i <= m_nStateVariables + m_polynomialDegree; ++i) {
            basisSize *= i;
        }
        for (size_t i = 2; i <= m_polynomialDegree; ++i) {
            basisSize /= i;
        }
    }
    // Finally, the linear state variables are degree 1. We extend the basis as follows:
    // (1 * B_1 ... 1 * B_s) (L_1 * B_1 ... L_1 * B_s) ... (L_p * B_1 ... L_p * B_s)
    // (L_1 ... L_p are the linear SV, and B_1 ... B_s the basis computed earlier)
    // This is of size s * (p + 1).
    return (withLinearSV ? (m_nLinearStateVariables + 1) : 1) * basisSize;
}

void AMCEngine::computeBasis(Matrix<double>& svMatrix, Matrix<double>& linearSVMatrix) {
    const size_t basisSize = getBasisSize(false);
    if (!m_useCrossTerms || m_nStateVariables <= 1) {
        // Computes (1, X_1^1 ... X_1^n, X_2^1 ... X_2^n, ... , X_m^1 ... X_m^n)
        for (size_t i = 0; i < m_nPaths; ++i) {
            auto basisRow = m_basis[i];
            assert(basisRow);
            const auto* svRow = svMatrix[i];
            assert(svRow);
            basisRow[0] = 1.0;
            for (size_t j = 0; j < m_nStateVariables; ++j) {
                const double x = svRow[j];
                double x_k = x;
                size_t offset = j * m_polynomialDegree;
                for (size_t k = 1; k <= m_polynomialDegree; ++k) {
                    basisRow[offset + k] = x_k;
                    x_k *= x;
                }
            }
        }
    }
    else {
        for (size_t i = 0; i < m_nPaths; ++i) {
            // We precompute (1 X_1 ... X_1^n) ... (1 X_m ... X_m^n) into m_monomialsPerStateVariable
            // This is very similar to the basis computation without cross terms, but "1" is repeated multiple
            // times for code readability.
            const auto* svRow = svMatrix[i];
            assert(svRow);
            for (size_t j = 0; j < m_nStateVariables; ++j) {
                const double x = svRow[j];
                double x_k = x;
                auto monomialsRow = m_monomialsPerStateVariable[j];
                assert(monomialsRow);
                monomialsRow[0] = 1.0;
                for (size_t k = 1; k <= m_polynomialDegree; ++k) {
                    monomialsRow[k] = x_k;
                    x_k *= x;
                }
            }
            // Then, we use the precomputed matrix and exponents to compute the basis :
            // \prod{S_k^\alpha_k} = \prod{m_monomialsPerStateVariable[k][\alpha_k]}
            auto basisRow = m_basis[i];
            assert(basisRow);
            for (size_t j = 0; j < basisSize; ++j) {
                auto monomialsExponentRow = m_monomialExponents[j];
                assert(monomialsExponentRow);
                basisRow[j] = 1.0;
                for (size_t k = 0; k < m_nStateVariables; ++k) {
                    basisRow[j] *= m_monomialsPerStateVariable[k][monomialsExponentRow[k]];
                }
            }
        }
    }
    if (!m_nLinearStateVariables)
        return;
    // Expand the basis, as explained in getBasisSize().
    // (1 * B_1 ... 1 * B_s) (L_1 * B_1 ... L_1 * B_s) ... (L_p * B_1 ... L_p * B_s)
    for (size_t i = 0; i < m_nPaths; ++i) {
        const auto* linearSVRow = linearSVMatrix[i];
        assert(linearSVRow);
        auto basisRow = m_basis[i];
        assert(basisRow);
        for (size_t j = 0; j < m_nLinearStateVariables; ++j) {
            const double x = linearSVRow[j];
            size_t offset = (j + 1) * basisSize;
            for (size_t k = 0; k < basisSize; ++k) {
                basisRow[offset + k] = basisRow[k] * x;
            }
        }
    }
}

void AMCEngine::updateFutureFlows(const size_t exIdx) {
    // We rescale the current exercise rebate by the exercise decision.
    for (size_t j = 0; j < m_nPaths; ++j) {
        m_exerciseFlows[exIdx].scaleAmount(j, m_exerciseDecision[j]);
    }
    // Future exercise rebates are scaled by the 'alive probability', that is (1 - exercise decision).
    for (size_t nextEx = exIdx + 1; nextEx < m_nExercises; ++nextEx) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_exerciseFlows[nextEx].scaleAmount(j, 1.0 - m_exerciseDecision[j]);
        }
    }
    // Future contract flows are also scaled by the 'alive probability'.
    // If a flow is included in the rebate, and is observed at the exercise observation date, its exercise index is exIdx 
    // => thus, it is not rescaled as it will be paid regardless of exercise.
    // If a flow is not included in the rebate and observed at the exercise observation date, its exercise index is exIdx + 1
    // => thus, it is rescaled as it will NOT be paid in case of exercise.
    // Contract flows with Exercise Index == (m_nExercises + 1) are bullet (paid regardless of an exercise event being triggered), so not rescaled.
    for (auto& flow : m_contractFlows) {
        if (flow.getExerciseIndex() > exIdx && flow.getExerciseIndex() != m_nExercises + 1) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                flow.scaleAmount(j, 1.0 - m_exerciseDecision[j]);
            }
        }
    }
}

void AMCEngine::rescaleByWeights(const size_t exIdx) {
    // We are rescaling the trajectories by weights in order to give more or less importance 
    // to some paths.
    // We are then solving the linear system (tX W^2 X)^-1 * (tX W^2 Y). (please note the square here !)
    // Note that W is a diagonal *positive* matrix, with W being the weights.
    // the solution to the linear regression becomes min_p(W*X*p - W*Y) = min_p(\sum_i w_i^2 (<X_i,p> - y_i)^2)
    m_exercises[exIdx]->computeWeights(m_weights, m_performances[exIdx]);
    for (size_t j = 0; j < m_nPaths; ++j) {
        m_conditionalExpectation[j] *= m_weights[j];
    }
    // Here, we need to keep track of the basis before rescaling, as the regressed conditional expectation is X*p.
    const size_t basisSize = m_basis.getNbCols();
    for (size_t j = 0; j < m_nPaths; ++j) {
        auto basisWeightedRow = m_basisWeighted[j];
        assert(basisWeightedRow);
        const auto* basisRow = m_basis[j];
        assert(basisRow);
        const double w = m_weights[j];
        for (size_t k = 0; k < basisSize; ++k) {
            basisWeightedRow[k] = basisRow[k] * w;
        }
    }
}

bool AMCEngine::needRegression(const size_t exIdx) {
    // We are not computing a regression for AMCExercise_NoExercise.
    if (m_exercises[exIdx]->isNoExercise())
        return false;
    const Time exDate = m_exerciseFlows[exIdx].getObservationDate();
    // We are checking whether there are (future) contract flows are known strictly after the exercise *observation* date.
    for (const auto& flow : m_contractFlows) {
        const size_t flowIdx = flow.getExerciseIndex();
        if (flowIdx > exIdx && flowIdx != m_nExercises + 1 && flow.getObservationDate() > exDate)
            return true;
    }
    // We are doing the same for the (future) exercises. Note we also discard AMCExercise_NoExercise.
    for (size_t nextEx = exIdx + 1; nextEx < m_nExercises; ++nextEx) {
        if (!m_exercises[nextEx]->isNoExercise() && m_exerciseFlows[nextEx].getObservationDate() > exDate)
            return true;
    }
    // All flows are known at the exercise date, we do not regress.
    return false;
}

void AMCEngine::clampConditionalExpectation(const double minimumGain, const double maximumGain) {
    // This function is clamping the regressed conditional expectation by the minimum and maximum realized trajectories.
    // This is done to stabilize the regression for lower density zones.
    for (size_t j = 0; j < m_nPaths; ++j) {
        m_conditionalExpectation[j] = std::max(minimumGain, std::min(m_conditionalExpectation[j], maximumGain));
    }
}

void AMCEngine::computeConditionalExpectation(const size_t exIdx) {
    // m_conditionalExpectation is the conditional expectation of (m_premiumBefore - rebate) as we regress on the exercise gain.
    double minimumGain = DBL_MAX, maximumGain = -DBL_MAX;
    for (size_t j = 0; j < m_nPaths; ++j) {
        const double gain = m_premiumBefore[j] - (m_exerciseFlows[exIdx].getAmount(j) * m_exerciseFlows[exIdx].getDFObsToSettleDate(j));
        m_conditionalExpectation[j] = gain;
        minimumGain = std::min(minimumGain, gain);
        maximumGain = std::max(maximumGain, gain);
    }
    // Check if we need to solve the linear regression - otherwise, just take the realized value as conditional expectation.
    if (!needRegression(exIdx))
        return;
    // We rescale the trajectories by a weighting vector. This is done to give more importance to certain trajectories,
    // and discard other useless paths.
    rescaleByWeights(exIdx);
    // Solve the linear system using SVD. m_conditionalExpectation now contains the regressed values.
    solveLinearRegression_SVD(m_basisWeighted, m_conditionalExpectation);
    productMatrixVector(m_basis, m_conditionalExpectation);
    // For stability, we clamp the conditional expectation by the minimum and maximum realized values.
    clampConditionalExpectation(minimumGain, maximumGain);
}

void AMCEngine::computePremiumBeforeExercise(const size_t exIdx) {
    // The premium (before exercise) contains:
    // 1- The premium at the next exercise (exIdx + 1), discounted to the current exercise observation date.
    if (exIdx + 1 < m_nExercises) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_premiumBefore[j] = m_premiumAfter[j] * m_exerciseFlows[exIdx + 1].getDFToObservationDate(j) / m_exerciseFlows[exIdx].getDFToObservationDate(j);
        }
    }
    // 2- The sum of contract flows with exercise index (exIdx + 1 - i.e. the continuation flows, not paid in case of exercise at exIdx), 
    // also discounted to the current exercise observation date.
    for (const auto& flow : m_contractFlows) {
        if (flow.getExerciseIndex() == exIdx + 1) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_premiumBefore[j] += flow.getAmount(j) * flow.getDFToSettlementDate(j) / m_exerciseFlows[exIdx].getDFToObservationDate(j);
            }
        }
    }
    /*if (m_exerciseFlows[exIdx].getObservationDate() == m_exerciseFlows[exIdx - 1].getObservationDate()) {
        m_optimize = true;
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_rhs[j] = m_premiumBefore[j];
        }
    }*/
}

void AMCEngine::computePremiumAfterExercise(const size_t exIdx) {
    // After exercise, we have the premium being a linear combination of the rebate flow (discounted at the exercise observation date) and
    // the continuation flows.
    for (size_t j = 0; j < m_nPaths; ++j) {
        m_premiumAfter[j] =
            m_exerciseDecision[j] * (m_exerciseFlows[exIdx].getAmount(j) * m_exerciseFlows[exIdx].getDFObsToSettleDate(j))
            + (1.0 - m_exerciseDecision[j]) * m_premiumBefore[j];
    }
    // For Callable and Putable exercises, we make the additional check that the premium should decrease (resp. increase) after the exercise.
    // If it is not the case, we revert the exercise decision.
    const bool isCallable = m_exercises[exIdx]->isCallable(), isPutable = m_exercises[exIdx]->isPutable();
    if (isCallable || isPutable) {
        double premiumBeforeEx = 0.0, premiumAfterEx = 0.0;
        for (size_t j = 0; j < m_nPaths; ++j) {
            premiumBeforeEx += m_premiumBefore[j];
            premiumAfterEx += m_premiumAfter[j];
        }
        premiumBeforeEx /= double(m_nPaths);
        premiumAfterEx /= double(m_nPaths);
        // We made things worse : do not exercise at this period.
        if ((isCallable && premiumAfterEx > premiumBeforeEx) || (isPutable && premiumAfterEx < premiumBeforeEx)) {
            for (size_t j = 0; j < m_nPaths; ++j) {
                m_exerciseDecision[j] = 0.0;
                m_premiumAfter[j] = m_premiumBefore[j];
            }
        }
    }
    // Finally, it is possible that we cannot exercise the whole option at once. The exercise decision is then capped by m_exercisableProportion.
    if (m_exercisableProportion < 1.0) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_exerciseDecision[j] = std::min(m_exercisableProportion, m_exerciseDecision[j]);
            m_premiumAfter[j] =
                m_exerciseDecision[j] * (m_exerciseFlows[exIdx].getAmount(j) * m_exerciseFlows[exIdx].getDFObsToSettleDate(j))
                + (1.0 - m_exerciseDecision[j]) * m_premiumBefore[j];
        }
    }
}

void AMCEngine::computeIndicators(const size_t exIdx) {
    // Specific indicators for AMCEngine.
    double premiumBeforeEx = 0.0, premiumAfterEx = 0.0, exitProba = 0.0;
    for (size_t j = 0; j < m_nPaths; ++j) {
        premiumBeforeEx += m_premiumBefore[j];
        premiumAfterEx += m_premiumAfter[j];
        exitProba += m_exerciseDecision[j];
    }
    premiumBeforeEx /= static_cast<double>(m_nPaths);
    premiumAfterEx /= static_cast<double>(m_nPaths);
    exitProba /= static_cast<double>(m_nPaths);
    m_exitImpact[exIdx] = premiumAfterEx - premiumBeforeEx;
    m_exitProbability[exIdx] = exitProba;
    for (size_t nextEx = exIdx + 1; nextEx < m_nExercises; ++nextEx) {
        m_exitProbability[nextEx] *= (1.0 - exitProba);
    }
}

void AMCEngine::setFlowExerciseIndex(AMCFlow& flow) const {
    size_t exerciseIndex = 0;
    // Specific treatment of bullet flows so they are paid regardless of exercise events.
    if (flow.isBulletFlow()) {
        exerciseIndex = m_nExercises + 1;
    }
    // For flows included in the rebate, we are allocating so that exIdx(flow) = i <=> exDate[i - 1] < obs(flow) <= exDate[i]
    // This way, the flow will be allocated to the current exIdx, and paid regardless of the exercise decision.
    else if (flow.isIncludedInRebate()) {
        exerciseIndex = static_cast<size_t>(std::distance(
            m_exerciseFlows.cbegin(),
            std::lower_bound(
                m_exerciseFlows.cbegin(), m_exerciseFlows.cend(), flow,
                [](AMCFlow const& a, AMCFlow const& b) -> bool {
                    return a.getObservationDate() < b.getObservationDate();
                }
        )));
    }
    // For flows not included in the rebate, we are allocating so that exIdx(flow) = i + 1 <=> exDate[i] <= obs(flow) < exDate[i + 1],
    // so it is effectively paid if we did not exercise (and not paid if we exercise).
    else {
        exerciseIndex = static_cast<size_t>(std::distance(
            m_exerciseFlows.cbegin(),
            std::upper_bound(
                m_exerciseFlows.cbegin(), m_exerciseFlows.cend(), flow,
                [](AMCFlow const& a, AMCFlow const& b) -> bool {
                    return a.getObservationDate() < b.getObservationDate();
                }
        )));
    }
    flow.setExerciseIndex(exerciseIndex);
}

Matrix<double> random(25000, 361);
AMCEngine::AMCEngine() {
    // Engine data
    m_nPaths = random.getNbRows();
    assert(m_nPaths);
    m_polynomialDegree = 4;
    m_useCrossTerms = true;
    m_exercisableProportion = 1.0;

    // Contract data
    m_modelDate = Time{0};
    m_nExercises = 12;
    m_exercises.resize(m_nExercises);
    m_exerciseFlows.resize(m_nExercises);
    for (size_t i = 0; i < m_nExercises; ++i) {
        m_exerciseFlows[i] =
            AMCFlow(
                m_nPaths, i,
                Time(30 * (i + 1)), Time(30 * (i + 1) + 7), // Observation date and settlement date
                true,
                false);
    }
    
    m_smoothingWidth = 0.05;
    m_smoothingGearing = 1.0;
    for (size_t i = 0; i < m_nExercises; ++i) {
        auto smoothingParamsPtr = createSmoothingParameters(
            UnderlyingType::Mono,
            random.getNbRows(),
            { 1000.0 }, // deltaMax
            { 0.03 },   // epsMin
            { 0.10 },   // epsMax
            { 0.80 },   // Barrier
            { 1.00 },   // FX
            -1000.0,    // Notional
            m_smoothingGearing,
            BarrierType::UpBarrier,
            m_modelDate,
            m_exerciseFlows[i].getObservationDate());
        double smoothingWidth = getSmoothingWidth(i);
        m_exercises[i] = std::make_shared<AMCExercise_ConditionalPutable>(smoothingParamsPtr, smoothingWidth);
    }

    // Various initializations
    m_conditionalExpectation.resize(m_nPaths, 0.0);
    m_premiumBefore.resize(m_nPaths, 0.0);
    m_premiumAfter.resize(m_nPaths, 0.0);
    m_exerciseDecision.resize(m_nPaths, 0.0);
    m_weights.resize(m_nPaths, 1.0);
    m_exitProbability.resize(m_nExercises, 0.0);
    m_exitImpact.resize(m_nExercises, 0.0);
    m_nStateVariables = 1;
    m_nLinearStateVariables = 0;
    m_monomialsPerStateVariable = Matrix<double>(m_nStateVariables, m_polynomialDegree + 1);
}

double getDiscountFactor(Time t) {
    return exp(-0.05 * t / 365.0);
}

void initRng() {
    size_t nValues = random.getNbRows() * random.getNbCols();
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 5489);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, nValues, random.data(), 0.0, sqrt(1.0 / 365.0));
}

double getPerformance(size_t iPath, Time time) {
    double r = 0.05;
    double q = 0.02;
    double sigma = 0.25;
    // (r - q) * t / 365 + sigma * W_t
    double t = double(time) / 365.0;
    double W_t = 0.0;
    for (int i = 0; i < int(time); ++i) {
        W_t += random[iPath][i];
    }
    return exp((r - q - 0.5 * sigma * sigma) * t + sigma * W_t);
}

double getTV(
    const size_t nPaths,
    std::vector<AMCFlow> const& contractFlows,
    std::vector<AMCFlow> const& rebateFlows) 
{
    double TV = 0.0;
    for (const auto& flow : contractFlows) {
        for (size_t i = 0; i < nPaths; ++i)
            TV += flow.getAmount(i) * flow.getDFToSettlementDate(i);
    }
    for (const auto& flow : rebateFlows) {
        for (size_t i = 0; i < nPaths; ++i)
            TV += flow.getAmount(i) * flow.getDFToSettlementDate(i);
    }
    return TV / double(nPaths);
}

void AMCEngine::computeForward() {
    initRng();

    // Exercise flows = Monthly callable at 100%
    m_exerciseFlows.resize(m_nExercises);
    for (size_t i = 0; i < m_nExercises; ++i) {
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_exerciseFlows[i].setAmount(j, 1.00 + 0.01 * (i + 1), 0.0, 0.0);
            m_exerciseFlows[i].setDiscountFactors(j,
                getDiscountFactor(m_exerciseFlows[i].getSettlementDate() - m_modelDate),
                getDiscountFactor(m_exerciseFlows[i].getSettlementDate() - m_exerciseFlows[i].getObservationDate())
            );
            
        }
    }

    // Performances
    m_performances.resize(m_nExercises);
    for (size_t i = 0; i < m_nExercises; ++i) {
        m_performances[i] = Matrix<double>(m_nPaths, 1); // Single underlying.
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_performances[i][j][0] = getPerformance(j, m_exerciseFlows[i].getObservationDate());
        }
    }

    // Contract flows.
    const size_t nFlows = 26;
    m_contractFlows.resize(nFlows);
    for (size_t i = 0; i < nFlows - 1; ++i) {
        m_contractFlows[i] =
            AMCFlow(
                m_nPaths, 0,
                Time(std::min(int(15 * i), 360)), Time(15 * i + 7), // Observation date and settlement date
                true,
                false);
        setFlowExerciseIndex(m_contractFlows[i]);
        for (size_t j = 0; j < m_nPaths; ++j) {
            m_contractFlows[i].setAmount(j, getPerformance(j, m_contractFlows[i].getObservationDate()) >= 1.0 ? 0.004 : 0.0, 0.0, 0.0);
            m_contractFlows[i].setDiscountFactors(j,
                getDiscountFactor(m_contractFlows[i].getSettlementDate() - m_modelDate),
                getDiscountFactor(m_contractFlows[i].getSettlementDate() - m_contractFlows[i].getObservationDate())
            );
        }
    }
    // Terminal flow = ZCB at 1Y.
    size_t i = nFlows - 1;
    m_contractFlows[i] =
        AMCFlow(
            m_nPaths, 0,
            Time{ 360 }, Time{ 367 }, // Observation date and settlement date
            false,                    // This flow is not included in the rebate.
            false);
    setFlowExerciseIndex(m_contractFlows[i]);
    for (size_t j = 0; j < m_nPaths; ++j) {
        const double perf = getPerformance(j, m_contractFlows[i].getObservationDate());
        m_contractFlows[i].setAmount(j, perf >= 0.6 ? (perf >= 1.0 ? perf : 1.0) : perf, 0.0, 0.0);
        m_contractFlows[i].setDiscountFactors(j,
            getDiscountFactor(m_contractFlows[i].getSettlementDate() - m_modelDate),
            getDiscountFactor(m_contractFlows[i].getSettlementDate() - m_contractFlows[i].getObservationDate())
        );
    }

    // m_performances
    m_stateVariables = m_performances;
    // No linear state variable.
    m_linearStateVariables.resize(m_nExercises);
    m_nStateVariables = 1;
    m_nLinearStateVariables = 0;
    m_basis = Matrix<double>(m_nPaths, getBasisSize(true));
    m_basisWeighted = Matrix<double>(m_nPaths, getBasisSize(true));
    m_monomialExponents = Matrix<size_t>(getBasisSize(false), m_nStateVariables);
    precomputeMonomialExponents();
}

void AMCEngine::computeBackward() {
    // The backward loop.
    size_t exIdx = m_nExercises - 1;
    while (exIdx < m_nExercises && m_exerciseFlows[exIdx].getObservationDate() >= m_modelDate) {
        // This takes the premium vector from the next exercise premium (= m_premiumAfter), discount it to the current exercise observation date,
        // and adds the period flows (discounted to the same date) to the premium vector.
        computePremiumBeforeExercise(exIdx);
        // Rescale state variables.
        rescaleStateVariable(m_stateVariables[exIdx]);
        rescaleStateVariable(m_linearStateVariables[exIdx]);
        // Compute the basis.
        computeBasis(m_stateVariables[exIdx], m_linearStateVariables[exIdx]);
        // Compute the conditional expectation.
        computeConditionalExpectation(exIdx);
        // Compute the exercise decision
        m_exercises[exIdx]->computeExercise(m_exerciseDecision, m_conditionalExpectation, m_performances[exIdx]);
        // Update premiumAfter = exDecision * m_exerciseFlows (= rebate premium) + (1 - exDecision) * premiumBefore (= continuation premium)  
        computePremiumAfterExercise(exIdx);
        // Update the new flows : 
        // The current exercise is geared by exDecision, the future exercises are geared by (1 - exDecision) = the alive proportion.
        // The future payoff flows are geared by (1 - exDecision), the current payoff flow is not geared (as the flow is included in the rebate)
        updateFutureFlows(exIdx);
        // Compute the AMC-specific indicators
        computeIndicators(exIdx);
        // Decrement the exercise index.
        exIdx--;
    }
}
