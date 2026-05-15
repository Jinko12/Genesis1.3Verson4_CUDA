#ifndef __GENESIS_CUDA_KERNELS__
#define __GENESIS_CUDA_KERNELS__

#include <complex>
#include <cstddef>

#include "Particle.h"

namespace genesis_cuda {

struct State;
struct BeamState;


struct BeamSliceDiagnostic {
    double x1;
    double x2;
    double y1;
    double y2;
    double px1;
    double px2;
    double py1;
    double py2;
    double g1;
    double g2;
    double xpx;
    double ypy;
    double xmin;
    double xmax;
    double pxmin;
    double pxmax;
    double ymin;
    double ymax;
    double pymin;
    double pymax;
    double gmin;
    double gmax;
    int count;
};

struct FieldSliceDiagnostic {
    double power;
    double x1;
    double x2;
    double y1;
    double y2;
    double ffRe;
    double ffIm;
    double centerRe;
    double centerIm;
    double fpower;
    double fx1;
    double fx2;
    double fy1;
    double fy2;
};

struct CachedFFTFieldView {
    State *state {nullptr};
    unsigned int ngrid {0};
    std::size_t batchSize {0};
};

bool hasDevice();
const char *lastError();

// Optional Stage 3.9B memory/resource audit. These functions are no-ops unless
// GENESIS_CUDA_MEMORY_AUDIT=1 is set. They do not mutate simulation state.
void setMemoryAuditContext(int worldRank, int mpiSize);
bool memoryAuditEnabled();
void printMemoryAuditSummary();

State *create();
void destroy(State *state);

bool ensure(State *state, unsigned int ngrid, std::size_t maxParticles);
bool uploadCoefficients(State *state,
                        const std::complex<double> *c,
                        const std::complex<double> *cbet,
                        const std::complex<double> *cwet,
                        unsigned int ngrid);

bool clearSource(State *state, unsigned int ngrid);

bool buildSource(State *state,
                 const Particle *particles,
                 std::size_t npart,
                 unsigned int ngrid,
                 double gridmax,
                 double dgrid,
                 int harm,
                 double scale,
                 double undAx,
                 double undAy,
                 double undKx,
                 double undKy,
                 double undGradx,
                 double undGrady);

bool adiStep(State *state,
             std::complex<double> *field,
             unsigned int ngrid,
             std::complex<double> cstep);

// Batched cuFFT field solver support. The field/source arrays contain
// batchSize independent ngrid x ngrid complex grids laid out consecutively.
bool ensureFFT(State *state, unsigned int ngrid, std::size_t maxParticles, std::size_t batchSize);

bool uploadFFTPropagator(State *state,
                         const std::complex<double> *K2,
                         const std::complex<double> *sigmoid,
                         unsigned int ngrid);

bool uploadFFTFields(State *state,
                     const std::complex<double> *fields,
                     unsigned int ngrid,
                     std::size_t batchSize);

bool downloadFFTFields(State *state,
                       std::complex<double> *fields,
                       unsigned int ngrid,
                       std::size_t batchSize);

bool fftFieldApplySlippage(State *state,
                           unsigned int ngrid,
                           std::size_t batchSize,
                           int direction);

// Download only the boundary slice that must be exchanged during MPI
// slippage.  The slice is returned as interleaved real/imag doubles with
// length 2 * ngrid * ngrid.  direction > 0 sends the logical last slice;
// direction < 0 sends the logical first slice.
bool fftFieldDownloadSlippageSlice(State *state,
                                   double *hostBuffer,
                                   std::size_t hostDoubles,
                                   unsigned int ngrid,
                                   std::size_t batchSize,
                                   int direction);

// Apply a one-record slippage on the CUDA FFT field buffer and inject the
// received MPI boundary slice.  If zeroBoundary is true the boundary slice is
// replaced by zeros and hostBoundary may be nullptr.
bool fftFieldApplySlippageBoundary(State *state,
                                   const double *hostBoundary,
                                   std::size_t hostDoubles,
                                   unsigned int ngrid,
                                   std::size_t batchSize,
                                   int direction,
                                   bool zeroBoundary);

bool copyFFTFieldsToBeam(BeamState *beamState,
                         std::size_t dstOffset,
                         State *fftState,
                         unsigned int ngrid,
                         std::size_t batchSize);

// Bind a BeamState field slot directly to a CUDA FFT field buffer. This avoids
// the device-to-device copy from the field solver cache into BeamState::radField.
// The binding is valid only while the referenced FFT State remains alive and its
// field buffer is not reallocated. Callers rebind during each BeamSolver field
// upload phase, so stale pointers are not carried across topology changes.
bool beamBindFFTField(BeamState *beamState,
                      std::size_t fieldIndex,
                      State *fftState,
                      unsigned int ngrid,
                      std::size_t batchSize);

// Bind a BeamState field slot to the internally owned BeamState::radField buffer.
// Use this after uploading or copying field data into radField.
bool beamUseInternalFieldData(BeamState *beamState,
                              std::size_t fieldIndex,
                              std::size_t dstOffset);

bool clearFFTSource(State *state, unsigned int ngrid, std::size_t batchSize);

bool buildFFTSourceAt(State *state,
                      const Particle *particles,
                      std::size_t npart,
                      unsigned int ngrid,
                      std::size_t sourceOffset,
                      double gridmax,
                      double dgrid,
                      int harm,
                      double scale,
                      double undAx,
                      double undAy,
                      double undKx,
                      double undKy,
                      double undGradx,
                      double undGrady);

bool buildFFTSourceFromBeam(State *state,
                            const BeamState *beamState,
                            const double *sliceScale,
                            std::size_t nslice,
                            unsigned int ngrid,
                            double gridmax,
                            double dgrid,
                            int harm,
                            double undAx,
                            double undAy,
                            double undKx,
                            double undKy,
                            double undGradx,
                            double undGrady);

bool executeFFTPropagation(State *state,
                           unsigned int ngrid,
                           std::size_t batchSize,
                           double delz,
                           bool doFilter);

// GPU-resident beam/field data model used by TrackBeam and BeamSolver.
// Particles are stored as Structure-of-Arrays on the device. Radiation fields
// are staged as contiguous complex grids, one field/harmonic block after
// another, with slice order already adjusted to Genesis' Field::first offset.
BeamState *beamCreate();
void beamDestroy(BeamState *state);

bool beamEnsure(BeamState *state, std::size_t nslice, std::size_t npart);

bool beamUploadParticles(BeamState *state,
                         const double *gamma,
                         const double *theta,
                         const double *x,
                         const double *y,
                         const double *px,
                         const double *py,
                         const int *sliceStart,
                         const int *sliceCount,
                         const int *particleSlice,
                         std::size_t nslice,
                         std::size_t npart);

bool beamDownloadParticles(BeamState *state,
                           double *gamma,
                           double *theta,
                           double *x,
                           double *y,
                           double *px,
                           double *py,
                           std::size_t npart);

bool beamUploadEz(BeamState *state, const double *ez, std::size_t npart);
bool beamClearEz(BeamState *state, std::size_t npart);

bool beamUploadFields(BeamState *state,
                      const std::complex<double> *fields,
                      const std::size_t *fieldOffset,
                      const int *fieldNgrid,
                      const int *fieldNslices,
                      const int *fieldHarm,
                      const double *fieldGridmax,
                      const double *fieldDgrid,
                      const double *fieldRtmp,
                      std::size_t fieldGridPoints,
                      std::size_t nfield);

bool beamEnsureFields(BeamState *state,
                      std::size_t fieldGridPoints,
                      std::size_t nfield);

bool beamUploadFieldMetadata(BeamState *state,
                             const std::size_t *fieldOffset,
                             const int *fieldNgrid,
                             const int *fieldNslices,
                             const int *fieldHarm,
                             const double *fieldGridmax,
                             const double *fieldDgrid,
                             const double *fieldRtmp,
                             std::size_t nfield);

bool beamUploadFieldDataAt(BeamState *state,
                           std::size_t dstOffset,
                           const std::complex<double> *fields,
                           std::size_t fieldGridPoints);

bool beamTrackTransverse(BeamState *state,
                         std::size_t npart,
                         double delz,
                         double aw,
                         double qx,
                         double qy,
                         double xoff,
                         double yoff,
                         int modeX,
                         int modeY);

bool beamApplyCorrector(BeamState *state, std::size_t npart, double cx, double cy);

bool beamApplyChicaneMatrix(BeamState *state,
                            std::size_t npart,
                            const double *matrix4x4);

bool beamApplyR56(BeamState *state, std::size_t npart, double r56, double gamma0);


// Compact GPU diagnostics. These functions reduce the device-resident beam/field
// to O(nslice) host summaries, avoiding full D2H synchronization during standard
// diagnostics. They are intended for diagnostics only and do not mutate state.
bool beamComputeSliceDiagnostics(BeamState *state,
                                 int nharm,
                                 BeamSliceDiagnostic *hostSliceDiagnostics,
                                 std::complex<double> *hostBunching);

bool fftFieldComputeSliceDiagnostics(State *state,
                                     unsigned int ngrid,
                                     std::size_t batchSize,
                                     bool includeFftMoments,
                                     FieldSliceDiagnostic *hostSliceDiagnostics);

bool beamAdvanceLongitudinal(BeamState *state,
                             std::size_t npart,
                             std::size_t nfield,
                             double delz,
                             double xks,
                             double xku,
                             double aw,
                             double autophase,
                             double undAx,
                             double undAy,
                             double undKx,
                             double undKy,
                             double undGradx,
                             double undGrady);

} // namespace genesis_cuda

#endif // __GENESIS_CUDA_KERNELS__
