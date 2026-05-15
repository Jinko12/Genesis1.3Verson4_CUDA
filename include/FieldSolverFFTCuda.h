#ifndef __GENESIS_FIELDSOLVERFFT_CUDA__
#define __GENESIS_FIELDSOLVERFFT_CUDA__

#include <complex>
#include <vector>

#ifdef FFTW
#include <fftw3.h>
#endif

#include "FieldSolver.h"
#include "GenesisCudaKernels.h"

class Field;
class Beam;
class Undulator;

class FieldSolverFFTCuda : public FieldSolver {
public:
    FieldSolverFFTCuda();
    ~FieldSolverFFTCuda() override;

    void init(double delz, double dgrid, double xks, unsigned int ngrid) override;
    void advance(double delz, Field *field, Beam *beam, Undulator *und) override;
    void initSourceFilter(double xc, double yc, double sig, bool doFilter) override;

#ifdef GENESIS_CUDA
    bool syncCudaFieldToHost(Field *field) override;
    void markCudaHostFieldDirty() override;
    bool copyCudaFieldToBeam(genesis_cuda::BeamState *beamState, const Field *field, std::size_t dstOffset) override;
    bool canBuildSourceFromCudaBeam() const override { return true; }
    bool applyCudaSlippage(Field *field, int direction) override;
    bool downloadCudaSlippageSlice(Field *field, int direction, double *hostBuffer, std::size_t hostDoubles) override;
    bool applyCudaSlippageBoundary(Field *field, int direction, const double *hostBoundary, std::size_t hostDoubles, bool zeroBoundary) override;
#endif

    static bool getCachedFieldView(const Field *field, genesis_cuda::CachedFFTFieldView *view);
    static void invalidateCachedFieldView(const Field *field);

private:
    void buildHostPropagator(double delz, double dgrid, double xks, unsigned int ngrid);
    bool prepareDevice(std::size_t maxParticles, std::size_t batchSize);
    bool shouldDeferHostDownload() const;
    bool stageHostFieldToDevice(Field *field, std::size_t nslice, std::size_t ngrid2);
    bool downloadDeviceFieldToHost(Field *field, std::size_t nslice, std::size_t ngrid2);
    bool buildSourceFromCudaBeam(Field *field, Beam *beam, Undulator *und, double delz, std::size_t nslice);
    void warnFallback(const char *reason);
    void registerCachedFieldView(Field *field, std::size_t batchSize);

    void advanceCpuRange(double delz, Field *field, Beam *beam, Undulator *und, unsigned long firstSlice);
    void fftHost(std::vector<std::complex<double>> &crfield);
    void releaseHostFFT();

    unsigned int ngrid_ {0};
    double delz_save_ {0.0};
    double dgrid_save_ {0.0};
    double xks_save_ {0.0};
    double dk_ {1.0};

    double xc_ {1.0};
    double yc_ {1.0};
    double sig_ {1.0};
    bool doFilter_ {false};

    std::complex<double> *in_ {nullptr};
    std::complex<double> *out_ {nullptr};
    bool hostPlanReady_ {false};
#ifdef FFTW
    fftw_plan p_ {nullptr};
    fftw_plan ip_ {nullptr};
#endif

    std::vector<std::complex<double>> uf_;
    std::vector<std::complex<double>> sf_;
    std::vector<std::complex<double>> K2_;
    std::vector<std::complex<double>> sigmoid_;
    std::vector<std::complex<double>> crsource_;
    std::vector<std::complex<double>> stagedFields_;
    std::vector<double> sliceScale_;

    genesis_cuda::State *cudaState_ {nullptr};
    bool cudaReady_ {false};
    bool propagatorUploaded_ {false};
    bool deviceFieldFresh_ {false};
    bool hostFieldFresh_ {true};
    std::size_t deviceBatchSize_ {0};
    bool warnedFallback_ {false};
    bool warnedNoFftFallback_ {false};
    const Field *registeredField_ {nullptr};
};

#endif // __GENESIS_FIELDSOLVERFFT_CUDA__
