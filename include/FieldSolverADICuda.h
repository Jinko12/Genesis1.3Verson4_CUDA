#ifndef __GENESIS_FIELDSOLVERADI_CUDA__
#define __GENESIS_FIELDSOLVERADI_CUDA__

#include <complex>
#include <vector>

#include "FieldSolver.h"
#include "GenesisCudaKernels.h"

class Field;
class Beam;
class Undulator;

class FieldSolverADICuda : public FieldSolver {
public:
    FieldSolverADICuda();
    ~FieldSolverADICuda() override;

    void init(double delz, double dgrid, double xks, unsigned int ngrid) override;
    void advance(double delz, Field *field, Beam *beam, Undulator *und) override;
    void initSourceFilter(double, double, double, bool) override;

private:
    void buildHostCoefficients(double delz, double dgrid, double xks, unsigned int ngrid);
    bool prepareDevice(std::size_t maxParticles);
    void warnFallback(const char *reason);

    void advanceCpuRange(double delz, Field *field, Beam *beam, Undulator *und, unsigned long firstSlice);
    void adiHost(std::vector<std::complex<double>> &crfield);
    void tridagxHost(std::vector<std::complex<double>> &u);
    void tridagyHost(std::vector<std::complex<double>> &u);

    unsigned int ngrid_ {0};
    double delz_save_ {0.0};
    std::complex<double> cstep_ {0.0, 0.0};

    std::vector<std::complex<double>> c_;
    std::vector<std::complex<double>> cbet_;
    std::vector<std::complex<double>> cwet_;
    std::vector<std::complex<double>> r_;
    std::vector<std::complex<double>> crsource_;

    genesis_cuda::State *cudaState_ {nullptr};
    bool coefficientsReady_ {false};
    bool cudaReady_ {false};
    bool warnedFallback_ {false};

};

inline void FieldSolverADICuda::initSourceFilter(double, double, double, bool) { }

#endif // __GENESIS_FIELDSOLVERADI_CUDA__
