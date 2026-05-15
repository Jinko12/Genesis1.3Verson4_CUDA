#ifndef __GENESIS_BEAMSOLVER__
#define __GENESIS_BEAMSOLVER__

#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <cstddef>

class Field;
class Beam;

#include "Undulator.h"
#include "EFieldSolver.h"
#include "TrackBeam.h"

#ifdef GENESIS_CUDA
namespace genesis_cuda { struct BeamState; }
#endif

using namespace std;

class BeamSolver {
public:
    BeamSolver();
    virtual ~BeamSolver();
    void initEField(double rmax, int ngrid, int nz, int nphi, double lambda, bool longr);
    void advance(double, Beam *, vector<Field *> *, Undulator *);
    void track(double, Beam *, Undulator *, bool);
    void applyR56(Beam *, Undulator *, double);
    double getSCField(int);
    void checkAllocation(unsigned long i);
    void setCudaTracking(bool enabled);
    bool cudaTrackingEnabled() const;
    void syncCudaTrackingToHost(Beam *beam);
    void invalidateCudaCache();
#ifdef GENESIS_CUDA
    genesis_cuda::BeamState *cudaBeamState();
    std::size_t cudaBeamParticleCount() const;
    bool cudaBeamStateAvailable() const;
#endif

private:
    complex<double> cpart;
    vector<double> rharm;
    vector<complex<double> > rpart;

    double ez{};
    double xks{}, xku{};

    double theta{}, gamma{}, btpar{};
    double k2gg{}, k2pp{}, k3gg{}, k3pp{};

    bool onlyFundamental;
    bool cudaTrackingRequested_ {false};
    bool warnedCudaTrackingFallback_ {false};

    void RungeKutta(double);
    void ODE(double, double);
    void advanceCpu(double, Beam *, vector<Field *> *, Undulator *);
    void warnCudaFallback(const char *reason);

    EFieldSolver efield;
    TrackBeam tracker;

#ifdef GENESIS_CUDA
    genesis_cuda::BeamState *cudaBeamState_ {nullptr};
    bool cudaBeamUploaded_ {false};
    bool cudaBeamDeviceDirty_ {false};
    std::size_t cudaBeamParticles_ {0};
    std::size_t cudaBeamSlices_ {0};

    vector<double> h_gamma_;
    vector<double> h_theta_;
    vector<double> h_x_;
    vector<double> h_y_;
    vector<double> h_px_;
    vector<double> h_py_;
    vector<double> h_ez_;
    vector<int> h_sliceStart_;
    vector<int> h_sliceCount_;
    vector<int> h_particleSlice_;

    vector<complex<double> > h_fieldBuffer_;
    vector<std::size_t> h_fieldOffset_;
    vector<int> h_fieldNgrid_;
    vector<int> h_fieldNslices_;
    vector<int> h_fieldHarm_;
    vector<double> h_fieldGridmax_;
    vector<double> h_fieldDgrid_;
    vector<double> h_fieldRtmp_;

    bool prepareCudaBeam(Beam *beam);
    bool downloadCudaBeam(Beam *beam);
    bool uploadCudaFields(vector<Field *> *field, const vector<int> &nfld, const vector<double> &rtmp);
    bool advanceCuda(double delz, Beam *beam, vector<Field *> *field, Undulator *und,
                     const vector<int> &nfld, const vector<double> &rtmp);
    std::size_t flattenBeam(Beam *beam);
#endif
};

inline double BeamSolver::getSCField(int islice){
    return efield.getSCField(islice);
}

inline void BeamSolver::initEField(double rmax, int ngrid, int nz, int nphi, double lambda, bool longr){
  efield.init(rmax,ngrid,nz,nphi,lambda,longr);
}

inline bool BeamSolver::cudaTrackingEnabled() const {
    return cudaTrackingRequested_;
}

#ifdef GENESIS_CUDA
inline genesis_cuda::BeamState *BeamSolver::cudaBeamState() {
    return cudaBeamUploaded_ ? cudaBeamState_ : nullptr;
}

inline std::size_t BeamSolver::cudaBeamParticleCount() const {
    return cudaBeamUploaded_ ? cudaBeamParticles_ : 0;
}

inline bool BeamSolver::cudaBeamStateAvailable() const {
    return cudaBeamUploaded_ && (cudaBeamState_ != nullptr);
}
#endif


#endif
