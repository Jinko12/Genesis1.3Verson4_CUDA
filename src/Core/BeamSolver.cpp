#include "BeamSolver.h"
#include "GenesisNvtx.h"
#include "Field.h"
#include "Beam.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#ifdef GENESIS_CUDA
#include "GenesisCudaKernels.h"
#include "FieldSolverFFTCuda.h"
#endif

BeamSolver::BeamSolver()
{
  onlyFundamental=false;
}

BeamSolver::~BeamSolver() {
#ifdef GENESIS_CUDA
  if (cudaBeamState_ != nullptr) {
    genesis_cuda::beamDestroy(cudaBeamState_);
    cudaBeamState_ = nullptr;
  }
#endif
}

void BeamSolver::warnCudaFallback(const char *reason) {
  if (!warnedCudaTrackingFallback_) {
    std::cerr << "[Genesis CUDA] Falling back to CPU beam tracking: " << reason << std::endl;
    warnedCudaTrackingFallback_ = true;
  }
}

void BeamSolver::setCudaTracking(bool enabled) {
#ifdef GENESIS_CUDA
  cudaTrackingRequested_ = enabled;
  if (!enabled) {
    cudaBeamUploaded_ = false;
    cudaBeamDeviceDirty_ = false;
  }
#else
  cudaTrackingRequested_ = false;
  if (enabled) {
    warnCudaFallback("this binary was built without USE_CUDA");
  }
#endif
}

void BeamSolver::invalidateCudaCache() {
#ifdef GENESIS_CUDA
  cudaBeamUploaded_ = false;
  cudaBeamDeviceDirty_ = false;
  cudaBeamParticles_ = 0;
  cudaBeamSlices_ = 0;
#endif
}

void BeamSolver::syncCudaTrackingToHost(Beam *beam) {
  GENESIS_NVTX_RANGE("beam.cuda.sync_to_host_if_dirty");
#ifdef GENESIS_CUDA
  if (!cudaTrackingRequested_ || !cudaBeamDeviceDirty_) {
    return;
  }
  if (!downloadCudaBeam(beam)) {
    std::cerr << "[Genesis CUDA] Cannot synchronize GPU beam state back to CPU: "
              << genesis_cuda::lastError() << std::endl;
    std::cerr << "[Genesis CUDA] Aborting to avoid continuing with stale particle coordinates." << std::endl;
    std::exit(EXIT_FAILURE);
  }
#else
  (void)beam;
#endif
}

#ifdef GENESIS_CUDA
std::size_t BeamSolver::flattenBeam(Beam *beam) {
  GENESIS_NVTX_RANGE("beam.cuda.flatten_host_beam");
  const std::size_t nslice = beam->beam.size();
  std::size_t total = 0;
  for (const auto &slice : beam->beam) {
    total += slice.size();
  }

  if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      nslice > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    warnCudaFallback("beam size exceeds CUDA beam metadata range");
    return total;
  }

  h_gamma_.resize(total);
  h_theta_.resize(total);
  h_x_.resize(total);
  h_y_.resize(total);
  h_px_.resize(total);
  h_py_.resize(total);
  h_ez_.resize(total);
  h_particleSlice_.resize(total);
  h_sliceStart_.resize(nslice);
  h_sliceCount_.resize(nslice);

  std::size_t pos = 0;
  for (std::size_t is = 0; is < nslice; ++is) {
    if (pos > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        beam->beam[is].size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      warnCudaFallback("slice particle offset/count exceeds CUDA beam metadata range");
      return total;
    }
    h_sliceStart_[is] = static_cast<int>(pos);
    h_sliceCount_[is] = static_cast<int>(beam->beam[is].size());
    for (std::size_t ip = 0; ip < beam->beam[is].size(); ++ip) {
      const Particle &p = beam->beam[is][ip];
      h_gamma_[pos] = p.gamma;
      h_theta_[pos] = p.theta;
      h_x_[pos] = p.x;
      h_y_[pos] = p.y;
      h_px_[pos] = p.px;
      h_py_[pos] = p.py;
      h_ez_[pos] = 0.0;
      h_particleSlice_[pos] = static_cast<int>(is);
      ++pos;
    }
  }
  return total;
}

bool BeamSolver::prepareCudaBeam(Beam *beam) {
  GENESIS_NVTX_RANGE("beam.cuda.prepare_beam");
  if (!cudaTrackingRequested_) {
    return false;
  }
  if (!genesis_cuda::hasDevice()) {
    warnCudaFallback(genesis_cuda::lastError());
    return false;
  }
  if (cudaBeamState_ == nullptr) {
    cudaBeamState_ = genesis_cuda::beamCreate();
    if (cudaBeamState_ == nullptr) {
      warnCudaFallback("failed to allocate CUDA beam state");
      return false;
    }
  }

  const std::size_t nslice = beam->beam.size();
  std::size_t total = 0;
  for (const auto &slice : beam->beam) {
    total += slice.size();
  }

  if (nslice > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    warnCudaFallback("beam slice count exceeds CUDA beam metadata range");
    return false;
  }

  if (cudaBeamUploaded_ &&
      (cudaBeamParticles_ == total) && (cudaBeamSlices_ == nslice)) {
    return true;
  }

  if (cudaBeamDeviceDirty_) {
    if (!downloadCudaBeam(beam)) {
      warnCudaFallback(genesis_cuda::lastError());
      return false;
    }
  }

  total = flattenBeam(beam);
  if (total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  if (!genesis_cuda::beamEnsure(cudaBeamState_, nslice, total)) {
    warnCudaFallback(genesis_cuda::lastError());
    return false;
  }
  if (!genesis_cuda::beamUploadParticles(cudaBeamState_,
                                         h_gamma_.data(),
                                         h_theta_.data(),
                                         h_x_.data(),
                                         h_y_.data(),
                                         h_px_.data(),
                                         h_py_.data(),
                                         h_sliceStart_.data(),
                                         h_sliceCount_.data(),
                                         h_particleSlice_.data(),
                                         nslice,
                                         total)) {
    warnCudaFallback(genesis_cuda::lastError());
    return false;
  }

  cudaBeamUploaded_ = true;
  cudaBeamDeviceDirty_ = false;
  cudaBeamParticles_ = total;
  cudaBeamSlices_ = nslice;
  return true;
}

bool BeamSolver::downloadCudaBeam(Beam *beam) {
  GENESIS_NVTX_RANGE("beam.cuda.download_particles");
  if (cudaBeamState_ == nullptr) {
    return true;
  }

  const std::size_t total = cudaBeamParticles_;
  h_gamma_.resize(total);
  h_theta_.resize(total);
  h_x_.resize(total);
  h_y_.resize(total);
  h_px_.resize(total);
  h_py_.resize(total);

  if (!genesis_cuda::beamDownloadParticles(cudaBeamState_,
                                           h_gamma_.data(),
                                           h_theta_.data(),
                                           h_x_.data(),
                                           h_y_.data(),
                                           h_px_.data(),
                                           h_py_.data(),
                                           total)) {
    return false;
  }

  if (beam->beam.size() != h_sliceStart_.size()) {
    warnCudaFallback("host beam slice count changed while CUDA beam state was dirty");
    return false;
  }

  for (std::size_t is = 0; is < beam->beam.size(); ++is) {
    const std::size_t start = static_cast<std::size_t>(h_sliceStart_[is]);
    const std::size_t count = static_cast<std::size_t>(h_sliceCount_[is]);
    if (beam->beam[is].size() != count || start + count > total) {
      warnCudaFallback("host beam particle layout changed while CUDA beam state was dirty");
      return false;
    }
    for (std::size_t ip = 0; ip < count; ++ip) {
      Particle &p = beam->beam[is][ip];
      const std::size_t idx = start + ip;
      p.gamma = h_gamma_[idx];
      p.theta = h_theta_[idx];
      p.x = h_x_[idx];
      p.y = h_y_[idx];
      p.px = h_px_[idx];
      p.py = h_py_[idx];
    }
  }

  cudaBeamDeviceDirty_ = false;
  cudaBeamUploaded_ = true;
  return true;
}

bool BeamSolver::uploadCudaFields(vector<Field *> *field, const vector<int> &nfld, const vector<double> &rtmp) {
  GENESIS_NVTX_RANGE("beam.cuda.bind_or_upload_fields");
  if (cudaBeamState_ == nullptr) {
    warnCudaFallback("CUDA beam state is not initialized");
    return false;
  }

  h_fieldBuffer_.clear();
  h_fieldOffset_.clear();
  h_fieldNgrid_.clear();
  h_fieldNslices_.clear();
  h_fieldHarm_.clear();
  h_fieldGridmax_.clear();
  h_fieldDgrid_.clear();
  h_fieldRtmp_.clear();

  const char *bindEnv = std::getenv("GENESIS_CUDA_BIND_FFT_FIELD");
  const bool bindFftFieldPointers = (bindEnv == nullptr) || (std::string(bindEnv) != "0");

  std::vector<Field *> fieldsToUpload;
  std::vector<genesis_cuda::CachedFFTFieldView> cachedFftViews;
  std::vector<bool> cachedFftUsable;
  std::vector<bool> bindExternally;
  fieldsToUpload.reserve(nfld.size());
  cachedFftViews.reserve(nfld.size());
  cachedFftUsable.reserve(nfld.size());
  bindExternally.reserve(nfld.size());

  // h_fieldOffset_ is meaningful only for internally owned BeamState::radField
  // data.  FFT-backed fields bound through fieldData[] ignore this offset and
  // read directly from the FieldSolverFFTCuda cache.
  std::size_t internalFieldGridPoints = 0;
  for (std::size_t j = 0; j < nfld.size(); ++j) {
    Field *fld = field->at(nfld[j]);
    const std::size_t ns = fld->field.size();
    const std::size_t ngrid = static_cast<std::size_t>(fld->ngrid);
    const std::size_t ngrid2 = ngrid * ngrid;
    const std::size_t fieldGridPoints = ns * ngrid2;

    if (ns > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        ngrid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      warnCudaFallback("field grid dimensions exceed CUDA beam metadata range");
      return false;
    }

    genesis_cuda::CachedFFTFieldView view;
    const bool cachedOk = FieldSolverFFTCuda::getCachedFieldView(fld, &view) &&
                          (view.ngrid == static_cast<unsigned int>(fld->ngrid)) &&
                          (view.batchSize == ns);
    const bool externalBinding = bindFftFieldPointers && cachedOk;

    h_fieldOffset_.push_back(externalBinding ? 0 : internalFieldGridPoints);
    h_fieldNgrid_.push_back(static_cast<int>(ngrid));
    h_fieldNslices_.push_back(static_cast<int>(ns));
    h_fieldHarm_.push_back(fld->getHarm());
    h_fieldGridmax_.push_back(fld->gridmax);
    h_fieldDgrid_.push_back(fld->dgrid);
    h_fieldRtmp_.push_back(rtmp[j]);
    fieldsToUpload.push_back(fld);
    cachedFftViews.push_back(view);
    cachedFftUsable.push_back(cachedOk);
    bindExternally.push_back(externalBinding);

    if (!externalBinding) {
      internalFieldGridPoints += fieldGridPoints;
    }
  }

  if (!genesis_cuda::beamEnsureFields(cudaBeamState_, internalFieldGridPoints, nfld.size())) {
    warnCudaFallback(genesis_cuda::lastError());
    return false;
  }

  if (!genesis_cuda::beamUploadFieldMetadata(cudaBeamState_,
                                             h_fieldOffset_.data(),
                                             h_fieldNgrid_.data(),
                                             h_fieldNslices_.data(),
                                             h_fieldHarm_.data(),
                                             h_fieldGridmax_.data(),
                                             h_fieldDgrid_.data(),
                                             h_fieldRtmp_.data(),
                                             nfld.size())) {
    warnCudaFallback(genesis_cuda::lastError());
    return false;
  }

  for (std::size_t j = 0; j < fieldsToUpload.size(); ++j) {
    Field *fld = fieldsToUpload[j];
    const std::size_t ns = fld->field.size();
    const std::size_t ngrid = static_cast<std::size_t>(fld->ngrid);
    const std::size_t ngrid2 = ngrid * ngrid;
    const std::size_t fieldGridPoints = ns * ngrid2;
    const std::size_t dstOffset = h_fieldOffset_[j];

    if (bindExternally[j]) {
      const genesis_cuda::CachedFFTFieldView &view = cachedFftViews[j];
      if (genesis_cuda::beamBindFFTField(cudaBeamState_,
                                         j,
                                         view.state,
                                         view.ngrid,
                                         view.batchSize)) {
        continue;
      }
      // The internal buffer was intentionally not allocated for this field.
      // Falling back here is safe because the beam has not been advanced yet;
      // the caller will use the CPU path after synchronizing field data.
      warnCudaFallback(genesis_cuda::lastError());
      return false;
    }

    if (cachedFftUsable[j]) {
      const genesis_cuda::CachedFFTFieldView &view = cachedFftViews[j];
      if (genesis_cuda::copyFFTFieldsToBeam(cudaBeamState_,
                                            dstOffset,
                                            view.state,
                                            view.ngrid,
                                            view.batchSize) &&
          genesis_cuda::beamUseInternalFieldData(cudaBeamState_, j, dstOffset)) {
        continue;
      }
    }

    if (!fld->syncCudaFieldToHost()) {
      warnCudaFallback("failed to synchronize CUDA field before host field upload");
      return false;
    }

    h_fieldBuffer_.resize(fieldGridPoints);
    for (std::size_t is = 0; is < ns; ++is) {
      const std::size_t src = (is + static_cast<std::size_t>(fld->first)) % ns;
      std::copy(fld->field[src].begin(),
                fld->field[src].end(),
                h_fieldBuffer_.begin() + is * ngrid2);
    }

    if (!genesis_cuda::beamUploadFieldDataAt(cudaBeamState_,
                                             dstOffset,
                                             h_fieldBuffer_.data(),
                                             fieldGridPoints)) {
      warnCudaFallback(genesis_cuda::lastError());
      return false;
    }
    if (!genesis_cuda::beamUseInternalFieldData(cudaBeamState_, j, dstOffset)) {
      warnCudaFallback(genesis_cuda::lastError());
      return false;
    }
  }
  return true;
}

bool BeamSolver::advanceCuda(double delz, Beam *beam, vector<Field *> *field, Undulator *und,
                             const vector<int> &nfld, const vector<double> &rtmp) {
  GENESIS_NVTX_RANGE("beam.cuda.advance_longitudinal");
  const bool spaceChargeOn = efield.hasShortRange() || efield.hasLongRange();

  if (spaceChargeOn) {
    GENESIS_NVTX_RANGE("beam.cuda.space_charge_pre_download");
    if (!downloadCudaBeam(beam)) {
      warnCudaFallback(genesis_cuda::lastError());
      return false;
    }
    cudaBeamUploaded_ = false;
  }

  if (!prepareCudaBeam(beam)) {
    return false;
  }

  if (spaceChargeOn) {
    GENESIS_NVTX_RANGE("beam.cuda.space_charge_cpu_field");
    const double aw = und->getaw();
    const double gammaz2 = und->getGammaRef() * und->getGammaRef() / (1.0 + aw * aw);
    efield.longRange(beam, und->getGammaRef(), aw);
    if (h_ez_.size() != cudaBeamParticles_) {
      h_ez_.assign(cudaBeamParticles_, 0.0);
    }
    for (std::size_t is = 0; is < beam->beam.size(); ++is) {
      const double eloss = -beam->longESC[is] / 511000.0;
      efield.shortRange(&beam->beam.at(is), beam->current.at(is), gammaz2, static_cast<int>(is));
      const std::size_t start = static_cast<std::size_t>(h_sliceStart_[is]);
      const std::size_t count = static_cast<std::size_t>(h_sliceCount_[is]);
      for (std::size_t ip = 0; ip < count; ++ip) {
        h_ez_[start + ip] = efield.getEField(ip) + eloss;
      }
    }
    if (!genesis_cuda::beamUploadEz(cudaBeamState_, h_ez_.data(), cudaBeamParticles_)) {
      warnCudaFallback(genesis_cuda::lastError());
      return false;
    }
  } else {
    if (!genesis_cuda::beamClearEz(cudaBeamState_, cudaBeamParticles_)) {
      warnCudaFallback(genesis_cuda::lastError());
      return false;
    }
  }

  {
    GENESIS_NVTX_RANGE("beam.cuda.field_binding_stage");
    if (!uploadCudaFields(field, nfld, rtmp)) {
      return false;
    }
  }

  const int undStep = und->getStep();
  const double ax = (undStep >= 0) ? und->ax[undStep] : 0.0;
  const double ay = (undStep >= 0) ? und->ay[undStep] : 0.0;
  const double kx = (undStep >= 0) ? und->kx[undStep] : 0.0;
  const double ky = (undStep >= 0) ? und->ky[undStep] : 0.0;
  const double gradx = (undStep >= 0) ? und->gradx[undStep] : 0.0;
  const double grady = (undStep >= 0) ? und->grady[undStep] : 0.0;

  {
    GENESIS_NVTX_RANGE("kernel.beam_longitudinal");
    if (!genesis_cuda::beamAdvanceLongitudinal(cudaBeamState_,
                                             cudaBeamParticles_,
                                             nfld.size(),
                                             delz,
                                             xks,
                                             xku,
                                             und->getaw(),
                                             und->autophase(),
                                             ax,
                                             ay,
                                             kx,
                                             ky,
                                             gradx,
                                             grady)) {
      std::cerr << "[Genesis CUDA] BeamSolver CUDA kernel failed after entering the mutating RK4 path: "
                << genesis_cuda::lastError() << std::endl;
      std::cerr << "[Genesis CUDA] Aborting to avoid double-advancing or using partially updated particles." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  cudaBeamDeviceDirty_ = true;
  cudaBeamUploaded_ = true;
  return true;
}
#endif

void BeamSolver::advance(double delz, Beam *beam, vector< Field *> *field, Undulator *und) {
    GENESIS_NVTX_RANGE("beam.solver.advance");

    vector<int> nfld;
    vector<double> rtmp;
    xks = 1;  // default value in the case that no field is defined

    for (int i = 0; i < field->size(); i++) {
        int harm = field->at(i)->getHarm();
        if ((harm == 1) || !onlyFundamental) {
            xks = field->at(i)->xks / static_cast<double>(harm);    // fundamental field wavenumber used in ODE below
            nfld.push_back(i);
            rtmp.push_back(und->fc(harm) / field->at(i)->xks);      // here the harmonics have to be taken care
        }
    }

    xku = und->getku();
    if (xku == 0) {
        xku = xks * 0.5 / und->getGammaRef() / und->getGammaRef();
    }

#ifdef GENESIS_CUDA
    if (cudaTrackingRequested_) {
        if (advanceCuda(delz, beam, field, und, nfld, rtmp)) {
            return;
        }
        syncCudaTrackingToHost(beam);
    }
    for (auto *fld : *field) {
        if ((fld != nullptr) && !fld->syncCudaFieldToHost()) {
            std::cerr << "[Genesis CUDA] Cannot synchronize CUDA field before CPU beam fallback: "
                      << genesis_cuda::lastError() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
#endif

    advanceCpu(delz, beam, field, und);
}

void BeamSolver::advanceCpu(double delz, Beam *beam, vector< Field *> *field, Undulator *und) {
    GENESIS_NVTX_RANGE("beam.cpu.advance_longitudinal");

    // here the harmonics needs to be taken into account

    vector<int> nfld;
    vector<double> rtmp;
    rpart.clear();
    rharm.clear();
    xks = 1;  // default value in the case that no field is defined

    for (int i = 0; i < field->size(); i++) {
        int harm = field->at(i)->getHarm();
        if ((harm == 1) || !onlyFundamental) {
            xks = field->at(i)->xks / static_cast<double>(harm);    // fundamental field wavenumber used in ODE below
            nfld.push_back(i);
            rtmp.push_back(und->fc(harm) / field->at(i)->xks);      // here the harmonics have to be taken care
            rpart.emplace_back(0);
            rharm.push_back(static_cast<double>(harm));
        }
    }

    xku = und->getku();
    if (xku ==
        0) {   // in the case of drifts - the beam stays in phase if it has the reference energy // this requires that the phase slippage is not applied
        xku = xks * 0.5 / und->getGammaRef() / und->getGammaRef();
    }

    double aw = und->getaw();
    double autophase = und->autophase();

    // obtaining long range space charge field
    efield.longRange(beam, und->getGammaRef(), aw);  // defines the array beam->longESC

    // Runge Kutta solver to advance particle
    auto gammaz2 = und->getGammaRef()*und->getGammaRef()/(1+aw*aw);
    for (int is = 0; is < beam->beam.size(); is++) {
        // accumulate space charge field
        double eloss = -beam->longESC[is] / 511000; // convert eV to units of electron rest mass
        efield.shortRange(&beam->beam.at(is), beam->current.at(is), gammaz2, is);
        for (int ip = 0; ip < beam->beam.at(is).size(); ip++) {
            gamma = beam->beam.at(is).at(ip).gamma;
            theta = beam->beam.at(is).at(ip).theta + autophase; // add autophase here
            double x = beam->beam.at(is).at(ip).x;
            double y = beam->beam.at(is).at(ip).y;
            double px = beam->beam.at(is).at(ip).px;
            double py = beam->beam.at(is).at(ip).py;
            double awloc = und->faw(x, y);                 // get the transverse dependence of the undulator field
            btpar = 1 + px * px + py * py + aw * aw * awloc * awloc;
            ez = efield.getEField(ip) + eloss;  // adding global long range space charge field to each particle
            cpart = 0;
            double wx, wy;
            int idx;
            for (int ifld = 0; ifld < nfld.size(); ifld++) {
                auto islice = (is + field->at(nfld[ifld])->first) % field->at(nfld[ifld])->field.size();

                if (field->at(nfld[ifld])->getLLGridpoint(x, y, &wx, &wy, &idx)) { // check whether particle is on grid
                    cpart = field->at(nfld[ifld])->field[islice].at(idx) * wx * wy;
                    idx++;
                    cpart += field->at(nfld[ifld])->field[islice].at(idx) * (1 - wx) * wy;
                    idx += field->at(nfld[ifld])->ngrid - 1;
                    cpart += field->at(nfld[ifld])->field[islice].at(idx) * wx * (1 - wy);
                    idx++;
                    cpart += field->at(nfld[ifld])->field[islice].at(idx) * (1 - wx) * (1 - wy);
                    rpart[ifld] = rtmp[ifld] * awloc * conj(cpart);
                } else {
                    rpart[ifld] = 0;
                }
            }
            this->RungeKutta(delz);

            beam->beam.at(is).at(ip).gamma = gamma;
            beam->beam.at(is).at(ip).theta = theta;
        }
    }
}

void BeamSolver::track(double dz, Beam *beam, Undulator *und, bool last) {
  GENESIS_NVTX_RANGE(last ? "beam.solver.track_transverse_last" : "beam.solver.track_transverse_first");
#ifdef GENESIS_CUDA
  if (cudaTrackingRequested_) {
    if (prepareCudaBeam(beam)) {
      {
        GENESIS_NVTX_RANGE("kernel.beam_transverse");
        if (!tracker.trackCuda(dz, beam, und, last, cudaBeamState_, cudaBeamParticles_)) {
        std::cerr << "[Genesis CUDA] TrackBeam CUDA kernel failed after entering the mutating path: "
                  << genesis_cuda::lastError() << std::endl;
        std::cerr << "[Genesis CUDA] Aborting to avoid using partially updated particles." << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      cudaBeamDeviceDirty_ = true;
      cudaBeamUploaded_ = true;
      return;
    }
    warnCudaFallback("CUDA beam state could not be prepared for TrackBeam");
  }
#endif
  tracker.track(dz,beam,und,last);
}

void BeamSolver::applyR56(Beam *beam, Undulator *und, double reflen) {
  GENESIS_NVTX_RANGE("beam.solver.r56");
#ifdef GENESIS_CUDA
  if (cudaTrackingRequested_) {
    if (prepareCudaBeam(beam)) {
      {
        GENESIS_NVTX_RANGE("kernel.beam_r56");
        if (!tracker.applyR56Cuda(beam, und, reflen, cudaBeamState_, cudaBeamParticles_)) {
        std::cerr << "[Genesis CUDA] R56 CUDA kernel failed after entering the mutating path: "
                  << genesis_cuda::lastError() << std::endl;
        std::cerr << "[Genesis CUDA] Aborting to avoid using partially updated particles." << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      cudaBeamDeviceDirty_ = true;
      cudaBeamUploaded_ = true;
      return;
    }
    warnCudaFallback("CUDA beam state could not be prepared for R56");
  }
#endif
  tracker.applyR56(beam,und,reflen);
}

void BeamSolver::RungeKutta(double delz) {
    // Runge Kutta Solver 4th order - taken from pushp from the old Fortran source


    // first step
    k2gg = 0;
    k2pp = 0;

    this->ODE(gamma, theta);

    // second step
    double stpz = 0.5 * delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    k3gg = k2gg;
    k3pp = k2pp;

    k2gg = 0;
    k2pp = 0;

    this->ODE(gamma, theta);

    // third step
    gamma += stpz * (k2gg - k3gg);
    theta += stpz * (k2pp - k3pp);

    k3gg /= 6;
    k3pp /= 6;

    k2gg *= -0.5;
    k2pp *= -0.5;

    this->ODE(gamma, theta);

    // fourth step
    stpz = delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    k3gg -= k2gg;
    k3pp -= k2pp;

    k2gg *= 2;
    k2pp *= 2;

    this->ODE(gamma, theta);
    gamma += stpz * (k3gg + k2gg / 6.0);
    theta += stpz * (k3pp + k2pp / 6.0);

}


void BeamSolver::ODE(double tgam,double tthet) {

    // differential equation for longitudinal motion
    double ztemp1 = -2. / xks;
    complex<double> ctmp = 0;
    for (int i = 0; i < rpart.size(); i++) {
        ctmp += rpart[i] * complex<double>(cos(rharm[i] * tthet), -sin(rharm[i] * tthet));
    }
    double btper0 = btpar + ztemp1 * ctmp.real();   //perpendicular velocity
    double btpar0 = sqrt(1. - btper0 / (tgam * tgam));     //parallel velocity
#ifdef G4_DBGDIAG
    // CL: detect negative radicands as NaN theta values can be the result
    double btpar0_sq=1.-btper0/(tgam*tgam);     //(parallel velocity)^2
    if(btpar0_sq<0) {
      cout << "DBGDIAG(BeamSolver::ODE): error, negative radicand detected" << endl;
    }
#endif
    k2pp += xks * (1. - 1. / btpar0) + xku;             //dtheta/dz
    k2gg += ctmp.imag() / btpar0 / tgam - ez;         //dgamma/dz
}

void BeamSolver::checkAllocation(unsigned long nslice) {
    efield.allocateForOutput(nslice);
}
