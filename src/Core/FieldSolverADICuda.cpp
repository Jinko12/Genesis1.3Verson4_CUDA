#include "FieldSolverADICuda.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "Beam.h"
#include "Field.h"
#include "Undulator.h"

using std::complex;
using std::size_t;
using std::vector;

FieldSolverADICuda::FieldSolverADICuda() = default;

FieldSolverADICuda::~FieldSolverADICuda() {
    if (cudaState_ != nullptr) {
        genesis_cuda::destroy(cudaState_);
        cudaState_ = nullptr;
    }
}

void FieldSolverADICuda::warnFallback(const char *reason) {
    if (!warnedFallback_) {
        std::cerr << "[Genesis CUDA] Falling back to CPU ADI field solver: " << reason << std::endl;
        warnedFallback_ = true;
    }
}

void FieldSolverADICuda::buildHostCoefficients(double delz, double dgrid, double xks, unsigned int ngrid) {
    delz_save_ = delz;
    ngrid_ = ngrid;

    const double rtmp = 0.25 * delz / (xks * dgrid * dgrid); // dz/(4 ks dx^2)
    cstep_ = complex<double>(0.0, rtmp);

    vector<double> mupp(ngrid);
    vector<double> mmid(ngrid);
    vector<double> mlow(ngrid);
    vector<complex<double>> cwrk1(ngrid);
    vector<complex<double>> cwrk2(ngrid);

    c_.resize(ngrid);
    cbet_.resize(ngrid);
    cwet_.resize(ngrid);
    r_.resize(static_cast<size_t>(ngrid) * ngrid);
    crsource_.resize(static_cast<size_t>(ngrid) * ngrid);

    mupp[0] = rtmp;
    mmid[0] = -2.0 * rtmp;
    mlow[0] = 0.0;
    for (unsigned int i = 1; i < ngrid - 1; ++i) {
        mupp[i] = rtmp;
        mmid[i] = -2.0 * rtmp;
        mlow[i] = rtmp;
    }
    mupp[ngrid - 1] = 0.0;
    mmid[ngrid - 1] = -2.0 * rtmp;
    mlow[ngrid - 1] = rtmp;

    for (unsigned int i = 0; i < ngrid; ++i) {
        cwrk1[i] = complex<double>(0.0, -mupp[i]);
        cwrk2[i] = complex<double>(1.0, -mmid[i]);
        c_[i] = complex<double>(0.0, -mlow[i]);
    }

    cbet_[0] = 1.0 / cwrk2[0];
    cwet_[0] = 0.0;
    for (unsigned int i = 1; i < ngrid; ++i) {
        cwet_[i] = cwrk1[i - 1] * cbet_[i - 1];
        cbet_[i] = 1.0 / (cwrk2[i] - c_[i] * cwet_[i]);
    }

    coefficientsReady_ = false;
}

bool FieldSolverADICuda::prepareDevice(std::size_t maxParticles) {
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

    if (!genesis_cuda::ensure(cudaState_, ngrid_, maxParticles)) {
        warnFallback(genesis_cuda::lastError());
        cudaReady_ = false;
        return false;
    }

    if (!coefficientsReady_) {
        if (!genesis_cuda::uploadCoefficients(cudaState_, c_.data(), cbet_.data(), cwet_.data(), ngrid_)) {
            warnFallback(genesis_cuda::lastError());
            cudaReady_ = false;
            return false;
        }
        coefficientsReady_ = true;
    }

    cudaReady_ = true;
    return true;
}

void FieldSolverADICuda::init(double delz, double dgrid, double xks, unsigned int ngrid) {
    if (ngrid < 2) {
        warnFallback("ngrid must be at least 2");
        cudaReady_ = false;
        return;
    }

    if ((delz != delz_save_) || (ngrid != ngrid_) || c_.empty()) {
        buildHostCoefficients(delz, dgrid, xks, ngrid);
    }

    // Particle-buffer sizing is done in advance(), where the beam is available.
    prepareDevice(0);
}

void FieldSolverADICuda::advance(double delz, Field *field, Beam *beam, Undulator *und) {
    size_t maxParticles = 0;
    for (const auto &slice : beam->beam) {
        maxParticles = std::max(maxParticles, slice.size());
    }

    if (!cudaReady_ || !prepareDevice(maxParticles)) {
        advanceCpuRange(delz, field, beam, und, 0);
        return;
    }

    const int harm = field->getHarm();
    const int undStep = und->getStep();
    const bool canBuildSource = und->inUndulator() && field->isEnabled() && ((harm % 2) == 1);

    for (unsigned long ii = 0; ii < field->field.size(); ++ii) { // ii is index for the beam
        if (!genesis_cuda::clearSource(cudaState_, ngrid_)) {
            warnFallback(genesis_cuda::lastError());
            cudaReady_ = false;
            advanceCpuRange(delz, field, beam, und, ii);
            return;
        }

        if (canBuildSource && !beam->beam[ii].empty()) {
            double scl = und->fc(harm) * vacimp * beam->current[ii] * field->xks * delz;
            scl /= 4.0 * eev * static_cast<double>(beam->beam[ii].size()) * field->dgrid * field->dgrid;

            const double ax = (undStep >= 0) ? und->ax[undStep] : 0.0;
            const double ay = (undStep >= 0) ? und->ay[undStep] : 0.0;
            const double kx = (undStep >= 0) ? und->kx[undStep] : 0.0;
            const double ky = (undStep >= 0) ? und->ky[undStep] : 0.0;
            const double gradx = (undStep >= 0) ? und->gradx[undStep] : 0.0;
            const double grady = (undStep >= 0) ? und->grady[undStep] : 0.0;

            if (!genesis_cuda::buildSource(cudaState_,
                                           beam->beam[ii].data(),
                                           beam->beam[ii].size(),
                                           ngrid_,
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
                advanceCpuRange(delz, field, beam, und, ii);
                return;
            }
        }

        const unsigned long i = (ii + field->first) % field->field.size(); // index for the field
        if (!genesis_cuda::adiStep(cudaState_, field->field[i].data(), ngrid_, cstep_)) {
            warnFallback(genesis_cuda::lastError());
            cudaReady_ = false;
            advanceCpuRange(delz, field, beam, und, ii);
            return;
        }
    }
}


void FieldSolverADICuda::advanceCpuRange(double delz, Field *field, Beam *beam, Undulator *und, unsigned long firstSlice) {
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
        adiHost(field->field[i]);
    }
}

void FieldSolverADICuda::adiHost(vector<complex<double>> &crfield) {
    int idx = 0;
    const int ngrid = static_cast<int>(ngrid_);

    for (idx = 0; idx < ngrid; ++idx) {
        r_[idx] = crsource_[idx] + crfield[idx] + cstep_ * (crfield[idx + ngrid] - 2.0 * crfield[idx]);
    }
    for (idx = ngrid; idx < ngrid * (ngrid - 1); ++idx) {
        r_[idx] = crsource_[idx] + crfield[idx] + cstep_ * (crfield[idx + ngrid] - 2.0 * crfield[idx] + crfield[idx - ngrid]);
    }
    for (idx = ngrid * (ngrid - 1); idx < ngrid * ngrid; ++idx) {
        r_[idx] = crsource_[idx] + crfield[idx] + cstep_ * (crfield[idx - ngrid] - 2.0 * crfield[idx]);
    }

    tridagxHost(crfield);

    for (int ix = 0; ix < ngrid * ngrid; ix += ngrid) {
        idx = ix;
        r_[idx] = crsource_[idx] + crfield[idx] + cstep_ * (crfield[idx + 1] - 2.0 * crfield[idx]);
        for (idx = ix + 1; idx < ix + ngrid - 1; ++idx) {
            r_[idx] = crsource_[idx] + crfield[idx] + cstep_ * (crfield[idx + 1] - 2.0 * crfield[idx] + crfield[idx - 1]);
        }
        idx = ix + ngrid - 1;
        r_[idx] = crsource_[idx] + crfield[idx] + cstep_ * (crfield[idx - 1] - 2.0 * crfield[idx]);
    }

    tridagyHost(crfield);
}

void FieldSolverADICuda::tridagxHost(vector<complex<double>> &u) {
    const int ngrid = static_cast<int>(ngrid_);
    for (int i = 0; i < ngrid * ngrid; i += ngrid) {
        u[i] = r_[i] * cbet_[0];
        for (int k = 1; k < ngrid; ++k) {
            u[k + i] = (r_[k + i] - c_[k] * u[k + i - 1]) * cbet_[k];
        }
        for (int k = ngrid - 2; k >= 0; --k) {
            u[k + i] -= cwet_[k + 1] * u[k + i + 1];
        }
    }
}

void FieldSolverADICuda::tridagyHost(vector<complex<double>> &u) {
    const int ngrid = static_cast<int>(ngrid_);
    for (int i = 0; i < ngrid; ++i) {
        u[i] = r_[i] * cbet_[0];
    }
    for (int k = 1; k < ngrid; ++k) {
        const int n = k * ngrid;
        for (int i = 0; i < ngrid; ++i) {
            u[n + i] = (r_[n + i] - c_[k] * u[n + i - ngrid]) * cbet_[k];
        }
    }
    for (int k = ngrid - 2; k >= 0; --k) {
        const int n = k * ngrid;
        for (int i = 0; i < ngrid; ++i) {
            u[n + i] -= cwet_[k + 1] * u[n + i + ngrid];
        }
    }
}
