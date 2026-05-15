#include "FieldSolverFFTCuda.h"
#include "GenesisNvtx.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include "Beam.h"
#include "Field.h"
#include "Undulator.h"

using std::complex;
using std::size_t;
using std::vector;

namespace {

struct CachedFieldRegistryEntry {
    genesis_cuda::State *state {nullptr};
    unsigned int ngrid {0};
    std::size_t batchSize {0};
    int first {0};
};

std::unordered_map<const Field *, CachedFieldRegistryEntry> g_cachedFftFields;

} // namespace

FieldSolverFFTCuda::FieldSolverFFTCuda() = default;

FieldSolverFFTCuda::~FieldSolverFFTCuda() {
    invalidateCachedFieldView(registeredField_);
    releaseHostFFT();
    if (cudaState_ != nullptr) {
        genesis_cuda::destroy(cudaState_);
        cudaState_ = nullptr;
    }
}

bool FieldSolverFFTCuda::getCachedFieldView(const Field *field, genesis_cuda::CachedFFTFieldView *view) {
    if (view != nullptr) {
        *view = genesis_cuda::CachedFFTFieldView{};
    }
    if (field == nullptr) {
        return false;
    }
    const auto it = g_cachedFftFields.find(field);
    if (it == g_cachedFftFields.end()) {
        return false;
    }

    const CachedFieldRegistryEntry &entry = it->second;
    if ((entry.state == nullptr) ||
        (entry.ngrid != static_cast<unsigned int>(field->ngrid)) ||
        (entry.batchSize != field->field.size()) ||
        (entry.first != field->first)) {
        g_cachedFftFields.erase(it);
        return false;
    }

    if (view != nullptr) {
        view->state = entry.state;
        view->ngrid = entry.ngrid;
        view->batchSize = entry.batchSize;
    }
    return true;
}

void FieldSolverFFTCuda::invalidateCachedFieldView(const Field *field) {
    if (field != nullptr) {
        g_cachedFftFields.erase(field);
    }
}

void FieldSolverFFTCuda::registerCachedFieldView(Field *field, std::size_t batchSize) {
    if ((field == nullptr) || (cudaState_ == nullptr) || !cudaReady_) {
        return;
    }
    registeredField_ = field;
    g_cachedFftFields[field] = CachedFieldRegistryEntry{cudaState_, ngrid_, batchSize, field->first};
}

void FieldSolverFFTCuda::releaseHostFFT() {
#ifdef FFTW
    if (p_ != nullptr) {
        fftw_destroy_plan(p_);
        p_ = nullptr;
    }
    if (ip_ != nullptr) {
        fftw_destroy_plan(ip_);
        ip_ = nullptr;
    }
#endif
    delete[] in_;
    delete[] out_;
    in_ = nullptr;
    out_ = nullptr;
    hostPlanReady_ = false;
}

void FieldSolverFFTCuda::warnFallback(const char *reason) {
    if (!warnedFallback_) {
        std::cerr << "[Genesis CUDA] Falling back from CUDA FFT field solver: " << reason << std::endl;
        warnedFallback_ = true;
    }
}

void FieldSolverFFTCuda::initSourceFilter(double xc, double yc, double sig, bool doFilter) {
    xc_ = xc;
    yc_ = yc;
    sig_ = sig;
    doFilter_ = doFilter && (xc_ > 0.0) && (yc_ > 0.0) && (sig_ > 0.0);
    propagatorUploaded_ = false;
}

bool FieldSolverFFTCuda::shouldDeferHostDownload() const {
    // The CUDA path is intended to keep the field resident on the device across
    // z-steps.  Host synchronization is performed explicitly by diagnostics,
    // field dumps, CPU fallback, and MPI/slippage code paths.  Set
    // GENESIS_CUDA_DEFER_FIELD_D2H=0 to force legacy eager downloads while
    // debugging numerical differences.
    const char *env = std::getenv("GENESIS_CUDA_DEFER_FIELD_D2H");
    return !((env != nullptr) && (env[0] == '0') && (env[1] == '\0'));
}

void FieldSolverFFTCuda::buildHostPropagator(double delz, double dgrid, double xks, unsigned int ngrid) {
    delz_save_ = delz;
    dgrid_save_ = dgrid;
    xks_save_ = xks;
    ngrid_ = ngrid;
    dk_ = 4.0 * std::asin(1.0) / (static_cast<double>(ngrid_) * dgrid_save_);

    const size_t n = static_cast<size_t>(ngrid_) * ngrid_;
    uf_.assign(n, complex<double>(0.0, 0.0));
    sf_.assign(n, complex<double>(0.0, 0.0));
    K2_.assign(n, complex<double>(0.0, 0.0));
    sigmoid_.assign(n, complex<double>(1.0, 0.0));
    crsource_.assign(n, complex<double>(0.0, 0.0));

    const double shift = -0.5 * static_cast<double>(ngrid_ - 1);
    for (unsigned int iy = 0; iy < ngrid_; ++iy) {
        const double dy = static_cast<double>(iy) + shift;
        const double y = (yc_ != 0.0) ? (dy / static_cast<double>(ngrid_) / yc_) : 0.0;
        for (unsigned int ix = 0; ix < ngrid_; ++ix) {
            const double dx = static_cast<double>(ix) + shift;
            const double x = (xc_ != 0.0) ? (dx / static_cast<double>(ngrid_) / xc_) : 0.0;
            const unsigned int iiy = (iy + (ngrid_ + 1) / 2) % ngrid_;
            const unsigned int iix = (ix + (ngrid_ + 1) / 2) % ngrid_;
            const size_t ii = static_cast<size_t>(iiy) * ngrid_ + iix;
            K2_[ii] = complex<double>(0.0, -(dx * dx + dy * dy) * dk_ * dk_ / (2.0 * xks_save_));
            if (doFilter_) {
                const double r = (std::sqrt(x * x + y * y) - 1.0) / sig_;
                sigmoid_[ii] = complex<double>(1.0 / (1.0 + std::exp(r)), 0.0);
            }
        }
    }

    propagatorUploaded_ = false;
    deviceFieldFresh_ = false;
    hostFieldFresh_ = true;
    deviceBatchSize_ = 0;
    invalidateCachedFieldView(registeredField_);

#ifdef FFTW
    releaseHostFFT();
    in_ = new complex<double>[n];
    out_ = new complex<double>[n];
    p_ = fftw_plan_dft_2d(static_cast<int>(ngrid_),
                          static_cast<int>(ngrid_),
                          reinterpret_cast<fftw_complex *>(in_),
                          reinterpret_cast<fftw_complex *>(out_),
                          FFTW_FORWARD,
                          FFTW_MEASURE);
    ip_ = fftw_plan_dft_2d(static_cast<int>(ngrid_),
                           static_cast<int>(ngrid_),
                           reinterpret_cast<fftw_complex *>(in_),
                           reinterpret_cast<fftw_complex *>(out_),
                           FFTW_BACKWARD,
                           FFTW_MEASURE);
    hostPlanReady_ = (p_ != nullptr) && (ip_ != nullptr);
#endif
}

bool FieldSolverFFTCuda::stageHostFieldToDevice(Field *field, std::size_t nslice, std::size_t ngrid2) {
    GENESIS_NVTX_RANGE("field.cuda.stage_host_field_to_device");
    stagedFields_.resize(ngrid2 * nslice);
    for (size_t ii = 0; ii < nslice; ++ii) {
        const size_t fieldIndex = (ii + static_cast<size_t>(field->first)) % nslice;
        std::copy(field->field[fieldIndex].begin(),
                  field->field[fieldIndex].end(),
                  stagedFields_.begin() + ii * ngrid2);
    }

    if (!genesis_cuda::uploadFFTFields(cudaState_, stagedFields_.data(), ngrid_, nslice)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        deviceFieldFresh_ = false;
        return false;
    }

    deviceFieldFresh_ = true;
    hostFieldFresh_ = true;
    deviceBatchSize_ = nslice;
    return true;
}

bool FieldSolverFFTCuda::downloadDeviceFieldToHost(Field *field, std::size_t nslice, std::size_t ngrid2) {
    GENESIS_NVTX_RANGE("field.cuda.download_device_field_to_host");
    if (!deviceFieldFresh_) {
        return true;
    }
    if ((cudaState_ == nullptr) || (deviceBatchSize_ != nslice)) {
        warnFallback("CUDA field download requested without a matching device field buffer");
        return false;
    }

    stagedFields_.resize(ngrid2 * nslice);
    if (!genesis_cuda::downloadFFTFields(cudaState_, stagedFields_.data(), ngrid_, nslice)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        invalidateCachedFieldView(field);
        return false;
    }

    for (size_t ii = 0; ii < nslice; ++ii) {
        const size_t fieldIndex = (ii + static_cast<size_t>(field->first)) % nslice;
        std::copy(stagedFields_.begin() + ii * ngrid2,
                  stagedFields_.begin() + (ii + 1) * ngrid2,
                  field->field[fieldIndex].begin());
    }

    hostFieldFresh_ = true;
    registerCachedFieldView(field, nslice);
    return true;
}

bool FieldSolverFFTCuda::syncCudaFieldToHost(Field *field) {
    GENESIS_NVTX_RANGE("field.cuda.sync_fft_field_to_host");
    if (hostFieldFresh_) {
        return true;
    }
    if (field == nullptr) {
        return false;
    }
    const size_t nslice = field->field.size();
    const size_t ngrid2 = static_cast<size_t>(ngrid_) * ngrid_;
    return downloadDeviceFieldToHost(field, nslice, ngrid2);
}

void FieldSolverFFTCuda::markCudaHostFieldDirty() {
    invalidateCachedFieldView(registeredField_);
    deviceFieldFresh_ = false;
    hostFieldFresh_ = true;
    deviceBatchSize_ = 0;
}

bool FieldSolverFFTCuda::copyCudaFieldToBeam(genesis_cuda::BeamState *beamState,
                                             const Field *field,
                                             std::size_t dstOffset) {
    if ((beamState == nullptr) || (field == nullptr) || !deviceFieldFresh_ ||
        (deviceBatchSize_ != field->field.size())) {
        return false;
    }
    return genesis_cuda::copyFFTFieldsToBeam(beamState,
                                             dstOffset,
                                             cudaState_,
                                             ngrid_,
                                             deviceBatchSize_);
}

bool FieldSolverFFTCuda::applyCudaSlippage(Field *field, int direction) {
    GENESIS_NVTX_RANGE("field.cuda.apply_slippage_shift");
    if ((field == nullptr) || (cudaState_ == nullptr) || !cudaReady_ || !deviceFieldFresh_) {
        return false;
    }
    if ((direction != 1) && (direction != -1)) {
        return false;
    }

    const std::size_t nslice = field->field.size();
    if ((nslice == 0) || (deviceBatchSize_ != nslice) ||
        (ngrid_ != static_cast<unsigned int>(field->ngrid))) {
        return false;
    }

    if (!genesis_cuda::fftFieldApplySlippage(cudaState_, ngrid_, nslice, direction)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        invalidateCachedFieldView(field);
        return false;
    }

    const int n = static_cast<int>(nslice);
    if (direction > 0) {
        field->first = (field->first + n - 1) % n;
    } else {
        field->first = (field->first + 1) % n;
    }

    deviceFieldFresh_ = true;
    hostFieldFresh_ = false;
    deviceBatchSize_ = nslice;
    registerCachedFieldView(field, nslice);
    return true;
}

bool FieldSolverFFTCuda::downloadCudaSlippageSlice(Field *field,
                                                     int direction,
                                                     double *hostBuffer,
                                                     std::size_t hostDoubles) {
    GENESIS_NVTX_RANGE("field.cuda.download_slippage_slice");
    if ((field == nullptr) || (hostBuffer == nullptr) || (cudaState_ == nullptr) ||
        !cudaReady_ || !deviceFieldFresh_) {
        return false;
    }
    if ((direction != 1) && (direction != -1)) {
        return false;
    }

    const std::size_t nslice = field->field.size();
    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid_) * ngrid_;
    if ((nslice == 0) || (deviceBatchSize_ != nslice) ||
        (ngrid_ != static_cast<unsigned int>(field->ngrid)) ||
        (hostDoubles < 2 * ngrid2)) {
        return false;
    }

    if (!genesis_cuda::fftFieldDownloadSlippageSlice(cudaState_,
                                                     hostBuffer,
                                                     hostDoubles,
                                                     ngrid_,
                                                     nslice,
                                                     direction)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        invalidateCachedFieldView(field);
        return false;
    }
    return true;
}

bool FieldSolverFFTCuda::applyCudaSlippageBoundary(Field *field,
                                                   int direction,
                                                   const double *hostBoundary,
                                                   std::size_t hostDoubles,
                                                   bool zeroBoundary) {
    GENESIS_NVTX_RANGE("field.cuda.apply_slippage_boundary_kernel");
    if ((field == nullptr) || (cudaState_ == nullptr) || !cudaReady_ || !deviceFieldFresh_) {
        return false;
    }
    if ((direction != 1) && (direction != -1)) {
        return false;
    }

    const std::size_t nslice = field->field.size();
    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid_) * ngrid_;
    if ((nslice == 0) || (deviceBatchSize_ != nslice) ||
        (ngrid_ != static_cast<unsigned int>(field->ngrid)) ||
        (!zeroBoundary && (hostBoundary == nullptr)) ||
        (!zeroBoundary && (hostDoubles < 2 * ngrid2))) {
        return false;
    }

    if (!genesis_cuda::fftFieldApplySlippageBoundary(cudaState_,
                                                     hostBoundary,
                                                     hostDoubles,
                                                     ngrid_,
                                                     nslice,
                                                     direction,
                                                     zeroBoundary)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        invalidateCachedFieldView(field);
        return false;
    }

    const int n = static_cast<int>(nslice);
    if (direction > 0) {
        field->first = (field->first + n - 1) % n;
    } else {
        field->first = (field->first + 1) % n;
    }

    deviceFieldFresh_ = true;
    hostFieldFresh_ = false;
    deviceBatchSize_ = nslice;
    registerCachedFieldView(field, nslice);
    return true;
}

bool FieldSolverFFTCuda::buildSourceFromCudaBeam(Field *field,
                                                 Beam *beam,
                                                 Undulator *und,
                                                 double delz,
                                                 std::size_t nslice) {
    GENESIS_NVTX_RANGE("kernel.source_deposition_soa");
    if ((field == nullptr) || (beam == nullptr) || (und == nullptr) || !beam->cudaBeamStateAvailable()) {
        return false;
    }

    const int harm = field->getHarm();
    if (!(und->inUndulator() && field->isEnabled() && ((harm % 2) == 1))) {
        return true;
    }
    if (beam->beam.size() != nslice) {
        return false;
    }

    sliceScale_.assign(nslice, 0.0);
    for (size_t ii = 0; ii < nslice; ++ii) {
        const std::size_t count = beam->beam[ii].size();
        if (count == 0) {
            continue;
        }
        double scl = und->fc(harm) * vacimp * beam->current[ii] * field->xks * delz;
        scl /= 4.0 * eev * static_cast<double>(count) * field->dgrid * field->dgrid;
        sliceScale_[ii] = scl;
    }

    const int undStep = und->getStep();
    const double ax = (undStep >= 0) ? und->ax[undStep] : 0.0;
    const double ay = (undStep >= 0) ? und->ay[undStep] : 0.0;
    const double kx = (undStep >= 0) ? und->kx[undStep] : 0.0;
    const double ky = (undStep >= 0) ? und->ky[undStep] : 0.0;
    const double gradx = (undStep >= 0) ? und->gradx[undStep] : 0.0;
    const double grady = (undStep >= 0) ? und->grady[undStep] : 0.0;

    return genesis_cuda::buildFFTSourceFromBeam(cudaState_,
                                                beam->cudaBeamState(),
                                                sliceScale_.data(),
                                                nslice,
                                                ngrid_,
                                                field->gridmax,
                                                field->dgrid,
                                                harm,
                                                ax,
                                                ay,
                                                kx,
                                                ky,
                                                gradx,
                                                grady);
}

bool FieldSolverFFTCuda::prepareDevice(std::size_t maxParticles, std::size_t batchSize) {
    GENESIS_NVTX_RANGE("field.cuda.prepare_device");
    if (!genesis_cuda::hasDevice()) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        return false;
    }

    if (cudaState_ == nullptr) {
        cudaState_ = genesis_cuda::create();
        if (cudaState_ == nullptr) {
            warnFallback(genesis_cuda::lastError());
            cudaReady_ = false;
            return false;
        }
    }

    if (!genesis_cuda::ensureFFT(cudaState_, ngrid_, maxParticles, batchSize)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        return false;
    }

    if (!propagatorUploaded_) {
        if (!genesis_cuda::uploadFFTPropagator(cudaState_, K2_.data(), sigmoid_.data(), ngrid_)) {
            warnFallback(genesis_cuda::lastError());
            cudaReady_ = false;
            return false;
        }
        propagatorUploaded_ = true;
    }

    cudaReady_ = true;
    return true;
}

void FieldSolverFFTCuda::init(double delz, double dgrid, double xks, unsigned int ngrid) {
    GENESIS_NVTX_RANGE("field.cuda.fft_init");
    if (ngrid < 2) {
        warnFallback("ngrid must be at least 2");
        cudaReady_ = false;
        return;
    }

    const bool mustRebuild = K2_.empty() ||
                             (dgrid != dgrid_save_) ||
                             (xks != xks_save_) ||
                             (ngrid != ngrid_);
    if (mustRebuild) {
        buildHostPropagator(delz, dgrid, xks, ngrid);
    } else {
        delz_save_ = delz;
    }
}

void FieldSolverFFTCuda::advance(double delz, Field *field, Beam *beam, Undulator *und) {
    GENESIS_NVTX_RANGE("field.cuda.fft_advance");
    const size_t nslice = field->field.size();
    const size_t n = static_cast<size_t>(ngrid_) * ngrid_;

    // Any attempt to advance the field is a mutation boundary.  The old device
    // view remains useful only until this solver starts a new field update.
    invalidateCachedFieldView(field);
    registeredField_ = field;

    size_t maxParticles = 0;
    for (const auto &slice : beam->beam) {
        maxParticles = std::max(maxParticles, slice.size());
    }

    if ((nslice == 0) || (n == 0)) {
        return;
    }

    if (!prepareDevice(maxParticles, nslice)) {
        if (!hostFieldFresh_) {
            if (!downloadDeviceFieldToHost(field, nslice, n)) {
                std::cerr << "[Genesis CUDA] Cannot synchronize CUDA FFT field before CPU fallback: "
                          << genesis_cuda::lastError() << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        if (beam != nullptr) {
            beam->syncCudaTrackingToHost();
        }
        advanceCpuRange(delz, field, beam, und, 0);
        hostFieldFresh_ = true;
        deviceFieldFresh_ = false;
        return;
    }

    if ((!deviceFieldFresh_) || (deviceBatchSize_ != nslice)) {
        if (!stageHostFieldToDevice(field, nslice, n)) {
            if (beam != nullptr) {
                beam->syncCudaTrackingToHost();
            }
            advanceCpuRange(delz, field, beam, und, 0);
            hostFieldFresh_ = true;
            deviceFieldFresh_ = false;
            return;
        }
    }

    {
        GENESIS_NVTX_RANGE("field.cuda.clear_source");
        if (!genesis_cuda::clearFFTSource(cudaState_, ngrid_, nslice)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        if (!hostFieldFresh_) {
            downloadDeviceFieldToHost(field, nslice, n);
        }
        if (beam != nullptr) {
            beam->syncCudaTrackingToHost();
        }
        advanceCpuRange(delz, field, beam, und, 0);
            std::exit(EXIT_FAILURE);
        }
    }

    const int harm = field->getHarm();
    const int undStep = und->getStep();
    const bool canBuildSource = und->inUndulator() && field->isEnabled() && ((harm % 2) == 1);

    const double ax = (undStep >= 0) ? und->ax[undStep] : 0.0;
    const double ay = (undStep >= 0) ? und->ay[undStep] : 0.0;
    const double kx = (undStep >= 0) ? und->kx[undStep] : 0.0;
    const double ky = (undStep >= 0) ? und->ky[undStep] : 0.0;
    const double gradx = (undStep >= 0) ? und->gradx[undStep] : 0.0;
    const double grady = (undStep >= 0) ? und->grady[undStep] : 0.0;

    bool sourceBuiltOnDeviceBeam = false;
    if (canBuildSource && (beam != nullptr) && beam->cudaBeamStateAvailable()) {
        if (!buildSourceFromCudaBeam(field, beam, und, delz, nslice)) {
            warnFallback(genesis_cuda::lastError());
            cudaReady_ = false;
            if (!hostFieldFresh_) {
                downloadDeviceFieldToHost(field, nslice, n);
            }
            if (beam != nullptr) {
                beam->syncCudaTrackingToHost();
            }
            advanceCpuRange(delz, field, beam, und, 0);
            hostFieldFresh_ = true;
            deviceFieldFresh_ = false;
            return;
        }
        sourceBuiltOnDeviceBeam = true;
    }

    if (canBuildSource && !sourceBuiltOnDeviceBeam) {
        if (beam != nullptr) {
            beam->syncCudaTrackingToHost();
        }
        for (size_t ii = 0; ii < nslice; ++ii) {
            if (beam->beam[ii].empty()) {
                continue;
            }

            double scl = und->fc(harm) * vacimp * beam->current[ii] * field->xks * delz;
            scl /= 4.0 * eev * static_cast<double>(beam->beam[ii].size()) * field->dgrid * field->dgrid;

            if (!genesis_cuda::buildFFTSourceAt(cudaState_,
                                                beam->beam[ii].data(),
                                                beam->beam[ii].size(),
                                                ngrid_,
                                                ii * n,
                                                field->gridmax,
                                                field->dgrid,
                                                harm,
                                                scl,
                                                ax,
                                                ay,
                                                kx,
                                                ky,
                                                gradx,
                                                grady)) {
                warnFallback(genesis_cuda::lastError());
                cudaReady_ = false;
                if (!hostFieldFresh_) {
                    downloadDeviceFieldToHost(field, nslice, n);
                }
                advanceCpuRange(delz, field, beam, und, 0);
                hostFieldFresh_ = true;
                deviceFieldFresh_ = false;
                return;
            }
        }
    }

    {
        GENESIS_NVTX_RANGE("kernel.fft_propagation_batched");
        if (!genesis_cuda::executeFFTPropagation(cudaState_, ngrid_, nslice, delz_save_, doFilter_)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        if (!hostFieldFresh_) {
            downloadDeviceFieldToHost(field, nslice, n);
        }
        if (beam != nullptr) {
            beam->syncCudaTrackingToHost();
        }
        advanceCpuRange(delz, field, beam, und, 0);
        hostFieldFresh_ = true;
        deviceFieldFresh_ = false;
            return;
        }
    }

    deviceFieldFresh_ = true;
    hostFieldFresh_ = false;
    deviceBatchSize_ = nslice;
    registerCachedFieldView(field, nslice);

    if (!shouldDeferHostDownload()) {
        if (!downloadDeviceFieldToHost(field, nslice, n)) {
            std::cerr << "[Genesis CUDA] Cannot synchronize CUDA FFT field state back to CPU: "
                      << genesis_cuda::lastError() << std::endl;
            std::cerr << "[Genesis CUDA] Aborting to avoid continuing with stale field data." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

void FieldSolverFFTCuda::advanceCpuRange(double delz, Field *field, Beam *beam, Undulator *und, unsigned long firstSlice) {
    GENESIS_NVTX_RANGE("field.cpu.fft_fallback_advance");
#ifndef FFTW
    if (!warnedNoFftFallback_) {
        std::cerr << "[Genesis CUDA] CUDA FFT failed and this build has no FFTW CPU fallback; "
                  << "aborting to avoid silently changing solver results." << std::endl;
        warnedNoFftFallback_ = true;
    }
    (void)delz;
    (void)field;
    (void)beam;
    (void)und;
    (void)firstSlice;
    std::exit(EXIT_FAILURE);
#else
    if (!hostPlanReady_) {
        if (!warnedNoFftFallback_) {
            std::cerr << "[Genesis CUDA] CUDA FFT failed and FFTW fallback plan is not available; "
                      << "aborting to avoid silently changing solver results." << std::endl;
            warnedNoFftFallback_ = true;
        }
        std::exit(EXIT_FAILURE);
    }

    const int harm = field->getHarm();
    const bool canBuildSource = und->inUndulator() && field->isEnabled() && ((harm % 2) == 1);

    for (unsigned long ii = firstSlice; ii < field->field.size(); ++ii) {
        std::fill(crsource_.begin(), crsource_.end(), complex<double>(0.0, 0.0));

        if (canBuildSource && !beam->beam[ii].empty()) {
            double scl = und->fc(harm) * vacimp * beam->current[ii] * field->xks * delz;
            scl /= 4.0 * eev * static_cast<double>(beam->beam[ii].size()) * field->dgrid * field->dgrid;
            double wx = 0.0;
            double wy = 0.0;
            int idx = 0;

            for (auto &particle : beam->beam.at(ii)) {
                const double x = particle.x;
                const double y = particle.y;
                const double theta = static_cast<double>(harm) * particle.theta;
                const double gamma = particle.gamma;

                if (field->getLLGridpoint(x, y, &wx, &wy, &idx)) {
                    const double part = std::sqrt(und->faw2(x, y)) * scl / gamma;
                    const complex<double> cpart(std::sin(theta) * part, std::cos(theta) * part);

                    double weight = wx * wy;
                    crsource_[idx] += weight * cpart;
                    weight = (1.0 - wx) * wy;
                    idx++;
                    crsource_[idx] += weight * cpart;
                    weight = wx * (1.0 - wy);
                    idx += static_cast<int>(ngrid_) - 1;
                    crsource_[idx] += weight * cpart;
                    weight = (1.0 - wx) * (1.0 - wy);
                    idx++;
                    crsource_[idx] += weight * cpart;
                }
            }
        }

        const unsigned long i = (ii + field->first) % field->field.size();
        fftHost(field->field[i]);
    }
    invalidateCachedFieldView(field);
    deviceFieldFresh_ = false;
    hostFieldFresh_ = true;
    deviceBatchSize_ = 0;
#endif
}

void FieldSolverFFTCuda::fftHost(vector<complex<double>> &crfield) {
#ifdef FFTW
    const size_t n = static_cast<size_t>(ngrid_) * ngrid_;
    for (size_t ii = 0; ii < n; ++ii) {
        in_[ii] = crfield[ii];
    }
    fftw_execute(p_);

    for (size_t ii = 0; ii < n; ++ii) {
        uf_[ii] = out_[ii];
        in_[ii] = crsource_[ii];
    }
    fftw_execute(p_);

    for (size_t ii = 0; ii < n; ++ii) {
        sf_[ii] = out_[ii];
    }
    if (doFilter_) {
        for (size_t ii = 0; ii < n; ++ii) {
            sf_[ii] *= sigmoid_[ii];
        }
    }

    for (size_t ii = 0; ii < n; ++ii) {
        in_[ii] = uf_[ii] * std::exp(K2_[ii] * delz_save_) + 2.0 * sf_[ii];
    }
    fftw_execute(ip_);

    const double norm = 1.0 / static_cast<double>(n);
    for (size_t ii = 0; ii < n; ++ii) {
        crfield[ii] = out_[ii] * norm;
    }
#else
    (void)crfield;
#endif
}
