#include "GenesisCudaKernels.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace genesis_cuda {

namespace {

struct GComplex {
    double x;
    double y;
};

static_assert(sizeof(std::complex<double>) == sizeof(GComplex),
              "std::complex<double> must be layout-compatible with two doubles for CUDA transfers");

thread_local std::string g_lastError;


struct MemoryAuditCategory {
    std::size_t currentBytes {0};
    std::size_t peakBytes {0};
    std::size_t totalAllocatedBytes {0};
    std::size_t totalFreedBytes {0};
    unsigned long long allocations {0};
    unsigned long long frees {0};
};

struct MemoryAuditAllocation {
    std::size_t bytes {0};
    std::string name;
    int device {-1};
};

struct MemoryAuditState {
    bool initialized {false};
    bool enabled {false};
    int worldRank {-1};
    int mpiSize {-1};
    int device {-1};
    std::size_t currentBytes {0};
    std::size_t peakBytes {0};
    std::size_t totalAllocatedBytes {0};
    std::size_t totalFreedBytes {0};
    unsigned long long allocations {0};
    unsigned long long frees {0};
    unsigned long long unknownFrees {0};
    unsigned long long cufftPlanCreates {0};
    unsigned long long cufftPlanDestroys {0};
    std::size_t cufftWorkspaceEstimateCurrent {0};
    std::size_t cufftWorkspaceEstimatePeak {0};
    std::unordered_map<const void *, MemoryAuditAllocation> active;
    std::map<std::string, MemoryAuditCategory> categories;
};

MemoryAuditState g_memoryAudit;

inline bool envEnabled(const char *name, bool defaultValue = false) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return defaultValue;
    }
    if ((std::strcmp(env, "0") == 0) || (std::strcmp(env, "false") == 0) ||
        (std::strcmp(env, "FALSE") == 0) || (std::strcmp(env, "off") == 0) ||
        (std::strcmp(env, "OFF") == 0) || (std::strcmp(env, "no") == 0) ||
        (std::strcmp(env, "NO") == 0)) {
        return false;
    }
    return true;
}

inline bool cudaSafeModeEnabled() {
    // Conservative CUDA fast-path switch.  This intentionally does not disable
    // CUDA itself, device binding, or the user's selected solver; it only turns
    // off optional optimized kernels/paths that may obscure debugging.
    static int enabled = []() {
        return (envEnabled("GENESIS_CUDA_SAFE_MODE", false) ||
                envEnabled("GENESIS_CUDA_CONSERVATIVE", false)) ? 1 : 0;
    }();
    return enabled != 0;
}

inline bool lazyParticleStagingEnabled() {
    static int enabled = []() {
        const char *env = std::getenv("GENESIS_CUDA_LAZY_PARTICLE_STAGING");
        if (env == nullptr || env[0] == '\0') {
            return 1;
        }
        return envEnabled("GENESIS_CUDA_LAZY_PARTICLE_STAGING", true) ? 1 : 0;
    }();
    return enabled != 0;
}

inline bool inplaceSlippageEnabled() {
    static int enabled = []() {
        if (cudaSafeModeEnabled()) {
            return 0;
        }
        return envEnabled("GENESIS_CUDA_INPLACE_SLIPPAGE", true) ? 1 : 0;
    }();
    return enabled != 0;
}

inline MemoryAuditState &memoryAudit() {
    if (!g_memoryAudit.initialized) {
        g_memoryAudit.initialized = true;
        g_memoryAudit.enabled = envEnabled("GENESIS_CUDA_MEMORY_AUDIT", false);
    }
    return g_memoryAudit;
}

inline std::string normalizeAllocationName(const char *name) {
    if (name == nullptr) {
        return "unknown";
    }
    std::string s(name);
    const std::string prefix = "cudaMalloc(";
    if (s.rfind(prefix, 0) == 0 && !s.empty() && s.back() == ')') {
        s = s.substr(prefix.size(), s.size() - prefix.size() - 1);
    }
    return s;
}

inline void recordAllocation(const void *ptr, std::size_t bytes, const char *name) {
    MemoryAuditState &audit = memoryAudit();
    if (!audit.enabled || ptr == nullptr || bytes == 0) {
        return;
    }
    int device = -1;
    cudaGetDevice(&device);
    const std::string key = normalizeAllocationName(name);
    audit.active[ptr] = MemoryAuditAllocation{bytes, key, device};
    audit.currentBytes += bytes;
    audit.peakBytes = std::max(audit.peakBytes, audit.currentBytes);
    audit.totalAllocatedBytes += bytes;
    audit.allocations++;
    audit.device = device;
    MemoryAuditCategory &cat = audit.categories[key];
    cat.currentBytes += bytes;
    cat.peakBytes = std::max(cat.peakBytes, cat.currentBytes);
    cat.totalAllocatedBytes += bytes;
    cat.allocations++;
}

inline void recordFree(const void *ptr) {
    MemoryAuditState &audit = memoryAudit();
    if (!audit.enabled || ptr == nullptr) {
        return;
    }
    auto it = audit.active.find(ptr);
    if (it == audit.active.end()) {
        audit.unknownFrees++;
        return;
    }
    const std::size_t bytes = it->second.bytes;
    const std::string key = it->second.name;
    audit.currentBytes = (audit.currentBytes >= bytes) ? (audit.currentBytes - bytes) : 0;
    audit.totalFreedBytes += bytes;
    audit.frees++;
    MemoryAuditCategory &cat = audit.categories[key];
    cat.currentBytes = (cat.currentBytes >= bytes) ? (cat.currentBytes - bytes) : 0;
    cat.totalFreedBytes += bytes;
    cat.frees++;
    audit.active.erase(it);
}

inline cudaError_t trackedCudaMalloc(void **ptr, std::size_t bytes, const char *name) {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err == cudaSuccess) {
        recordAllocation(*ptr, bytes, name);
    }
    return err;
}

inline cudaError_t trackedCudaFree(void *ptr) {
    recordFree(ptr);
    return cudaFree(ptr);
}


inline void recordCufftPlanCreate(std::size_t estimatedWorkspaceBytes) {
    MemoryAuditState &audit = memoryAudit();
    if (!audit.enabled) {
        return;
    }
    audit.cufftPlanCreates++;
    audit.cufftWorkspaceEstimateCurrent += estimatedWorkspaceBytes;
    audit.cufftWorkspaceEstimatePeak = std::max(audit.cufftWorkspaceEstimatePeak,
                                                audit.cufftWorkspaceEstimateCurrent);
}

inline void recordCufftPlanDestroy(std::size_t estimatedWorkspaceBytes) {
    MemoryAuditState &audit = memoryAudit();
    if (!audit.enabled) {
        return;
    }
    audit.cufftPlanDestroys++;
    audit.cufftWorkspaceEstimateCurrent = (audit.cufftWorkspaceEstimateCurrent >= estimatedWorkspaceBytes)
        ? (audit.cufftWorkspaceEstimateCurrent - estimatedWorkspaceBytes) : 0;
}

inline double bytesToMiB(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline const char *cufftResultString(cufftResult err) {
    switch (err) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
#if defined(CUFFT_INCOMPLETE_PARAMETER_LIST)
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
#endif
#if defined(CUFFT_INVALID_DEVICE)
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
#endif
#if defined(CUFFT_PARSE_ERROR)
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
#endif
#if defined(CUFFT_NO_WORKSPACE)
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
#endif
#if defined(CUFFT_NOT_IMPLEMENTED)
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
#endif
#if defined(CUFFT_LICENSE_ERROR)
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
#endif
#if defined(CUFFT_NOT_SUPPORTED)
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
#endif
        default: return "unknown cuFFT error";
    }
}

inline bool setError(const char *context, cudaError_t err) {
    if (err == cudaSuccess) {
        return true;
    }
    g_lastError = std::string(context) + ": " + cudaGetErrorString(err);
    return false;
}

inline bool setCufftError(const char *context, cufftResult err) {
    if (err == CUFFT_SUCCESS) {
        return true;
    }
    g_lastError = std::string(context) + ": " + cufftResultString(err);
    return false;
}

inline bool checkKernel(const char *context) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return setError(context, err);
    }
#ifdef GENESIS_CUDA_DEBUG_SYNC
    err = cudaDeviceSynchronize();
    return setError(context, err);
#else
    (void)context;
    return true;
#endif
}

inline bool fastKernelsEnabled() {
    static int enabled = []() {
        if (cudaSafeModeEnabled()) {
            return 0;
        }
        return envEnabled("GENESIS_CUDA_FAST_KERNELS", true) ? 1 : 0;
    }();
    return enabled != 0;
}

inline bool cacheLongitudinalInterpolationEnabled() {
    static int enabled = []() {
        if (cudaSafeModeEnabled()) {
            return 0;
        }
        return envEnabled("GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD", true) ? 1 : 0;
    }();
    return enabled != 0;
}

inline bool longitudinalAlgebraOptimizedEnabled() {
    static int enabled = []() {
        if (cudaSafeModeEnabled()) {
            return 0;
        }
        return envEnabled("GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT", true) ? 1 : 0;
    }();
    return enabled != 0;
}

inline bool symmetricTransverseEnabled() {
    static int enabled = []() {
        if (cudaSafeModeEnabled()) {
            return 0;
        }
        return envEnabled("GENESIS_CUDA_SYMMETRIC_TRANSVERSE", true) ? 1 : 0;
    }();
    return enabled != 0;
}


inline int kernelBlockSizeFromEnv(const char *name, int defaultValue) {
    const char *env = std::getenv(name);
    if ((env == nullptr) || (env[0] == '\0')) {
        return defaultValue;
    }
    char *end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if ((end == env) || (*end != '\0') || (parsed < 32) || (parsed > 1024) || ((parsed % 32) != 0)) {
        return defaultValue;
    }
    return static_cast<int>(parsed);
}

inline int kernelBlockSizeFromEnv(const char *primaryName,
                                  const char *compatName,
                                  int defaultValue) {
    const char *primary = std::getenv(primaryName);
    if ((primary != nullptr) && (primary[0] != '\0')) {
        return kernelBlockSizeFromEnv(primaryName, defaultValue);
    }
    return kernelBlockSizeFromEnv(compatName, defaultValue);
}

inline int beamLongitudinalBlockSize() {
    static int block = kernelBlockSizeFromEnv("GENESIS_CUDA_BEAM_LONGITUDINAL_BLOCK",
                                              "GENESIS_CUDA_BEAM_LONG_BLOCK",
                                              256);
    return block;
}

inline int beamTransverseBlockSize() {
    static int block = kernelBlockSizeFromEnv("GENESIS_CUDA_BEAM_TRANSVERSE_BLOCK",
                                              "GENESIS_CUDA_BEAM_TRANS_BLOCK",
                                              256);
    return block;
}

inline int sourceDepositionBlockSize() {
    static int block = kernelBlockSizeFromEnv("GENESIS_CUDA_SOURCE_DEPOSITION_BLOCK",
                                              "GENESIS_CUDA_SOURCE_BLOCK",
                                              256);
    return block;
}

__device__ inline double atomicAddDouble(double *address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__host__ __device__ inline GComplex makeComplex(double x, double y) {
    GComplex z {x, y};
    return z;
}

__host__ __device__ inline GComplex add(GComplex a, GComplex b) {
    return makeComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline GComplex sub(GComplex a, GComplex b) {
    return makeComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline GComplex mul(GComplex a, GComplex b) {
    return makeComplex(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}

__host__ __device__ inline GComplex scale(GComplex a, double s) {
    return makeComplex(a.x * s, a.y * s);
}

__host__ __device__ inline GComplex cexpComplex(GComplex z) {
    const double er = exp(z.x);
    return makeComplex(er * cos(z.y), er * sin(z.y));
}

__host__ __device__ inline void sincosPortable(double phase, double *s, double *c) {
#if defined(__CUDA_ARCH__)
    sincos(phase, s, c);
#else
    *s = sin(phase);
    *c = cos(phase);
#endif
}


__device__ inline void atomicAddComplex(GComplex *address, GComplex value, double weight) {
    atomicAddDouble(&(address->x), value.x * weight);
    atomicAddDouble(&(address->y), value.y * weight);
}

__global__ void clearComplexKernel(GComplex *data, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = makeComplex(0.0, 0.0);
    }
}

__global__ void scaleComplexKernel(GComplex *data, std::size_t n, double factor) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = scale(data[idx], factor);
    }
}

__global__ void fftFieldSlippageKernel(const GComplex *oldField,
                                       GComplex *field,
                                       std::size_t ngrid2,
                                       std::size_t batchSize,
                                       int direction) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = ngrid2 * batchSize;
    if (idx >= total) {
        return;
    }

    const std::size_t local = idx % ngrid2;
    const std::size_t slice = idx / ngrid2;
    if (direction > 0) {
        if (slice == 0) {
            field[idx] = makeComplex(0.0, 0.0);
        } else {
            field[idx] = oldField[(slice - 1) * ngrid2 + local];
        }
    } else {
        if (slice + 1 >= batchSize) {
            field[idx] = makeComplex(0.0, 0.0);
        } else {
            field[idx] = oldField[(slice + 1) * ngrid2 + local];
        }
    }
}

__global__ void fftFieldSlippageBoundaryKernel(const GComplex *oldField,
                                               const GComplex *boundary,
                                               GComplex *field,
                                               std::size_t ngrid2,
                                               std::size_t batchSize,
                                               int direction,
                                               int zeroBoundary) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = ngrid2 * batchSize;
    if (idx >= total) {
        return;
    }

    const std::size_t local = idx % ngrid2;
    const std::size_t slice = idx / ngrid2;
    if (direction > 0) {
        if (slice == 0) {
            field[idx] = (zeroBoundary != 0) ? makeComplex(0.0, 0.0) : boundary[local];
        } else {
            field[idx] = oldField[(slice - 1) * ngrid2 + local];
        }
    } else {
        if (slice + 1 >= batchSize) {
            field[idx] = (zeroBoundary != 0) ? makeComplex(0.0, 0.0) : boundary[local];
        } else {
            field[idx] = oldField[(slice + 1) * ngrid2 + local];
        }
    }
}

__global__ void fftFieldSlippageInPlaceKernel(GComplex *field,
                                               const GComplex *boundary,
                                               std::size_t ngrid2,
                                               std::size_t batchSize,
                                               int direction,
                                               int zeroBoundary) {
    const std::size_t local = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= ngrid2 || batchSize == 0) {
        return;
    }

    const GComplex boundaryValue = (zeroBoundary != 0 || boundary == nullptr)
                                       ? makeComplex(0.0, 0.0)
                                       : boundary[local];
    if (direction > 0) {
        for (std::size_t slice = batchSize - 1; slice > 0; --slice) {
            field[slice * ngrid2 + local] = field[(slice - 1) * ngrid2 + local];
        }
        field[local] = boundaryValue;
    } else {
        for (std::size_t slice = 0; slice + 1 < batchSize; ++slice) {
            field[slice * ngrid2 + local] = field[(slice + 1) * ngrid2 + local];
        }
        field[(batchSize - 1) * ngrid2 + local] = boundaryValue;
    }
}

__global__ void fftPropagateKernel(GComplex *field,
                                   GComplex *source,
                                   const GComplex *K2,
                                   const GComplex *sigmoid,
                                   std::size_t ngrid2,
                                   std::size_t total,
                                   double delz,
                                   int doFilter) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const std::size_t local = idx % ngrid2;
    GComplex sf = source[idx];
    if (doFilter) {
        sf = mul(sf, sigmoid[local]);
    }
    field[idx] = add(mul(field[idx], cexpComplex(scale(K2[local], delz))), scale(sf, 2.0));
}

__global__ void buildSourceKernel(const Particle *particles,
                                  std::size_t npart,
                                  GComplex *source,
                                  unsigned int ngrid,
                                  double gridmax,
                                  double dgrid,
                                  int harm,
                                  double scaleFactor,
                                  double undAx,
                                  double undAy,
                                  double undKx,
                                  double undKy,
                                  double undGradx,
                                  double undGrady) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    const Particle particle = particles[ip];
    const double x = particle.x;
    const double y = particle.y;

    if (!((x > -gridmax) && (x < gridmax) && (y > -gridmax) && (y < gridmax))) {
        return;
    }

    double wx0 = (x + gridmax) / dgrid;
    double wy0 = (y + gridmax) / dgrid;
    const int ix = static_cast<int>(floor(wx0));
    const int iy = static_cast<int>(floor(wy0));

    // Match Field::getLLGridpoint(): wx/wy are weights for the lower-left grid point.
    const double wx = 1.0 + floor(wx0) - wx0;
    const double wy = 1.0 + floor(wy0) - wy0;
    int idx = ix + iy * static_cast<int>(ngrid);

    const double dx = x - undAx;
    const double dy = y - undAy;
    const double faw2 = 1.0 + undKx * dx * dx + undKy * dy * dy + 2.0 * (undGradx * dx + undGrady * dy);
    const double part = sqrt(faw2) * scaleFactor / particle.gamma;
    const double theta = static_cast<double>(harm) * particle.theta;
    double stheta = 0.0;
    double ctheta = 1.0;
    sincosPortable(theta, &stheta, &ctheta);
    const GComplex cpart = makeComplex(stheta * part, ctheta * part);

    atomicAddComplex(&source[idx], cpart, wx * wy);
    idx += 1;
    atomicAddComplex(&source[idx], cpart, (1.0 - wx) * wy);
    idx += static_cast<int>(ngrid) - 1;
    atomicAddComplex(&source[idx], cpart, wx * (1.0 - wy));
    idx += 1;
    atomicAddComplex(&source[idx], cpart, (1.0 - wx) * (1.0 - wy));
}

__global__ void buildSourceFromSoAKernel(const double *gamma,
                                         const double *theta,
                                         const double *x,
                                         const double *y,
                                         const int *sliceStart,
                                         const int *sliceCount,
                                         const double *sliceScale,
                                         GComplex *source,
                                         unsigned int ngrid,
                                         std::size_t ngrid2,
                                         std::size_t nslice,
                                         double gridmax,
                                         double dgrid,
                                         int harm,
                                         double undAx,
                                         double undAy,
                                         double undKx,
                                         double undKy,
                                         double undGradx,
                                         double undGrady) {
    const std::size_t islice = static_cast<std::size_t>(blockIdx.x);
    if (islice >= nslice) {
        return;
    }

    const int start = sliceStart[islice];
    const int count = sliceCount[islice];
    const double scaleFactor = sliceScale[islice];
    if ((count <= 0) || (scaleFactor == 0.0)) {
        return;
    }

    GComplex *sliceSource = source + islice * ngrid2;
    for (int local = threadIdx.x; local < count; local += blockDim.x) {
        const int ip = start + local;
        const double px = x[ip];
        const double py = y[ip];

        if (!((px > -gridmax) && (px < gridmax) && (py > -gridmax) && (py < gridmax))) {
            continue;
        }

        double wx0 = (px + gridmax) / dgrid;
        double wy0 = (py + gridmax) / dgrid;
        const int ix = static_cast<int>(floor(wx0));
        const int iy = static_cast<int>(floor(wy0));
        const double wx = 1.0 + floor(wx0) - wx0;
        const double wy = 1.0 + floor(wy0) - wy0;
        int idx = ix + iy * static_cast<int>(ngrid);

        const double dx = px - undAx;
        const double dy = py - undAy;
        const double faw2 = 1.0 + undKx * dx * dx + undKy * dy * dy + 2.0 * (undGradx * dx + undGrady * dy);
        const double part = sqrt(faw2) * scaleFactor / gamma[ip];
        const double phase = static_cast<double>(harm) * theta[ip];
        double sphase = 0.0;
        double cphase = 1.0;
        sincosPortable(phase, &sphase, &cphase);
        const GComplex cpart = makeComplex(sphase * part, cphase * part);

        atomicAddComplex(&sliceSource[idx], cpart, wx * wy);
        idx += 1;
        atomicAddComplex(&sliceSource[idx], cpart, (1.0 - wx) * wy);
        idx += static_cast<int>(ngrid) - 1;
        atomicAddComplex(&sliceSource[idx], cpart, wx * (1.0 - wy));
        idx += 1;
        atomicAddComplex(&sliceSource[idx], cpart, (1.0 - wx) * (1.0 - wy));
    }
}


__global__ void beamMomentsDiagnosticKernel(const double *gamma,
                                            const double *theta,
                                            const double *x,
                                            const double *y,
                                            const double *px,
                                            const double *py,
                                            const int *sliceStart,
                                            const int *sliceCount,
                                            std::size_t nslice,
                                            BeamSliceDiagnostic *out) {
    const std::size_t islice = static_cast<std::size_t>(blockIdx.x);
    if (islice >= nslice) {
        return;
    }

    const int tid = threadIdx.x;
    const int nthread = blockDim.x;
    const int start = sliceStart[islice];
    const int count = sliceCount[islice];

    enum { NV = 22 };
    extern __shared__ double sdiag[];
    double *v[NV];
    for (int i = 0; i < NV; ++i) {
        v[i] = sdiag + static_cast<std::size_t>(i) * nthread;
    }

    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    double px1 = 0.0, px2 = 0.0, py1 = 0.0, py2 = 0.0;
    double g1 = 0.0, g2 = 0.0, xpx = 0.0, ypy = 0.0;
    double xmin = 1.0e5, xmax = -1.0e5;
    double pxmin = 1.0e5, pxmax = -1.0e5;
    double ymin = 1.0e5, ymax = -1.0e5;
    double pymin = 1.0e5, pymax = -1.0e5;
    double gmin = 1.0e7, gmax = 1.0;

    (void)theta;
    for (int local = tid; local < count; local += nthread) {
        const int ip = start + local;
        const double xi = x[ip];
        const double yi = y[ip];
        const double pxi = px[ip];
        const double pyi = py[ip];
        const double gi = gamma[ip];
        x1 += xi;
        x2 += xi * xi;
        y1 += yi;
        y2 += yi * yi;
        px1 += pxi;
        px2 += pxi * pxi;
        py1 += pyi;
        py2 += pyi * pyi;
        g1 += gi;
        g2 += gi * gi;
        xpx += xi * pxi;
        ypy += yi * pyi;
        xmin = fmin(xmin, xi);
        xmax = fmax(xmax, xi);
        pxmin = fmin(pxmin, pxi);
        pxmax = fmax(pxmax, pxi);
        ymin = fmin(ymin, yi);
        ymax = fmax(ymax, yi);
        pymin = fmin(pymin, pyi);
        pymax = fmax(pymax, pyi);
        gmin = fmin(gmin, gi);
        gmax = fmax(gmax, gi);
    }

    v[0][tid] = x1;    v[1][tid] = x2;
    v[2][tid] = y1;    v[3][tid] = y2;
    v[4][tid] = px1;   v[5][tid] = px2;
    v[6][tid] = py1;   v[7][tid] = py2;
    v[8][tid] = g1;    v[9][tid] = g2;
    v[10][tid] = xpx;  v[11][tid] = ypy;
    v[12][tid] = xmin; v[13][tid] = xmax;
    v[14][tid] = pxmin; v[15][tid] = pxmax;
    v[16][tid] = ymin; v[17][tid] = ymax;
    v[18][tid] = pymin; v[19][tid] = pymax;
    v[20][tid] = gmin; v[21][tid] = gmax;
    __syncthreads();

    for (int stride = nthread / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int i = 0; i < 12; ++i) {
                v[i][tid] += v[i][tid + stride];
            }
            v[12][tid] = fmin(v[12][tid], v[12][tid + stride]);
            v[13][tid] = fmax(v[13][tid], v[13][tid + stride]);
            v[14][tid] = fmin(v[14][tid], v[14][tid + stride]);
            v[15][tid] = fmax(v[15][tid], v[15][tid + stride]);
            v[16][tid] = fmin(v[16][tid], v[16][tid + stride]);
            v[17][tid] = fmax(v[17][tid], v[17][tid + stride]);
            v[18][tid] = fmin(v[18][tid], v[18][tid + stride]);
            v[19][tid] = fmax(v[19][tid], v[19][tid + stride]);
            v[20][tid] = fmin(v[20][tid], v[20][tid + stride]);
            v[21][tid] = fmax(v[21][tid], v[21][tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        BeamSliceDiagnostic d;
        d.x1 = v[0][0];    d.x2 = v[1][0];
        d.y1 = v[2][0];    d.y2 = v[3][0];
        d.px1 = v[4][0];   d.px2 = v[5][0];
        d.py1 = v[6][0];   d.py2 = v[7][0];
        d.g1 = v[8][0];    d.g2 = v[9][0];
        d.xpx = v[10][0];  d.ypy = v[11][0];
        d.xmin = v[12][0]; d.xmax = v[13][0];
        d.pxmin = v[14][0]; d.pxmax = v[15][0];
        d.ymin = v[16][0]; d.ymax = v[17][0];
        d.pymin = v[18][0]; d.pymax = v[19][0];
        d.gmin = v[20][0]; d.gmax = v[21][0];
        d.count = count;
        out[islice] = d;
    }
}

__global__ void beamBunchingDiagnosticKernel(const double *theta,
                                             const int *sliceStart,
                                             const int *sliceCount,
                                             std::size_t nslice,
                                             int nharm,
                                             GComplex *out) {
    const std::size_t islice = static_cast<std::size_t>(blockIdx.x);
    const int iharm = static_cast<int>(blockIdx.y);
    if ((islice >= nslice) || (iharm >= nharm)) {
        return;
    }
    const int tid = threadIdx.x;
    const int nthread = blockDim.x;
    extern __shared__ double sbunch[];
    double *sre = sbunch;
    double *sim = sbunch + nthread;

    const int start = sliceStart[islice];
    const int count = sliceCount[islice];
    double re = 0.0;
    double im = 0.0;
    const double harm = static_cast<double>(iharm + 1);
    for (int local = tid; local < count; local += nthread) {
        const double phase = harm * theta[start + local];
        re += cos(phase);
        im += sin(phase);
    }
    sre[tid] = re;
    sim[tid] = im;
    __syncthreads();
    for (int stride = nthread / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sre[tid] += sre[tid + stride];
            sim[tid] += sim[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[islice * static_cast<std::size_t>(nharm) + static_cast<std::size_t>(iharm)] = makeComplex(sre[0], sim[0]);
    }
}

__global__ void fftFieldDiagnosticKernel(const GComplex *field,
                                         unsigned int ngrid,
                                         std::size_t ngrid2,
                                         std::size_t nslice,
                                         FieldSliceDiagnostic *out) {
    const std::size_t islice = static_cast<std::size_t>(blockIdx.x);
    if (islice >= nslice) {
        return;
    }
    const int tid = threadIdx.x;
    const int nthread = blockDim.x;
    enum { NV = 7 };
    extern __shared__ double sfield[];
    double *v[NV];
    for (int i = 0; i < NV; ++i) {
        v[i] = sfield + static_cast<std::size_t>(i) * nthread;
    }

    const double shift = -0.5 * static_cast<double>(ngrid - 1);
    const GComplex *slice = field + islice * ngrid2;
    double power = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;
    double y1 = 0.0;
    double y2 = 0.0;
    double ffRe = 0.0;
    double ffIm = 0.0;

    for (std::size_t idx = static_cast<std::size_t>(tid); idx < ngrid2; idx += nthread) {
        const unsigned int iy = static_cast<unsigned int>(idx / ngrid);
        const unsigned int ix = static_cast<unsigned int>(idx - static_cast<std::size_t>(iy) * ngrid);
        const double dx = static_cast<double>(ix) + shift;
        const double dy = static_cast<double>(iy) + shift;
        const GComplex loc = slice[idx];
        const double wei = loc.x * loc.x + loc.y * loc.y;
        power += wei;
        x1 += dx * wei;
        x2 += dx * dx * wei;
        y1 += dy * wei;
        y2 += dy * dy * wei;
        ffRe += loc.x;
        ffIm += loc.y;
    }
    v[0][tid] = power;
    v[1][tid] = x1;
    v[2][tid] = x2;
    v[3][tid] = y1;
    v[4][tid] = y2;
    v[5][tid] = ffRe;
    v[6][tid] = ffIm;
    __syncthreads();
    for (int stride = nthread / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int i = 0; i < NV; ++i) {
                v[i][tid] += v[i][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        const std::size_t center = (ngrid2 - 1) / 2;
        const GComplex c = slice[center];
        FieldSliceDiagnostic d;
        d.power = v[0][0];
        d.x1 = v[1][0];
        d.x2 = v[2][0];
        d.y1 = v[3][0];
        d.y2 = v[4][0];
        d.ffRe = v[5][0];
        d.ffIm = v[6][0];
        d.centerRe = c.x;
        d.centerIm = c.y;
        d.fpower = 0.0;
        d.fx1 = 0.0;
        d.fx2 = 0.0;
        d.fy1 = 0.0;
        d.fy2 = 0.0;
        out[islice] = d;
    }
}

__global__ void fftFieldFarfieldDiagnosticKernel(const GComplex *fftField,
                                                 unsigned int ngrid,
                                                 std::size_t ngrid2,
                                                 std::size_t nslice,
                                                 FieldSliceDiagnostic *out) {
    const std::size_t islice = static_cast<std::size_t>(blockIdx.x);
    if (islice >= nslice) {
        return;
    }
    const int tid = threadIdx.x;
    const int nthread = blockDim.x;
    enum { NV = 5 };
    extern __shared__ double sfar[];
    double *v[NV];
    for (int i = 0; i < NV; ++i) {
        v[i] = sfar + static_cast<std::size_t>(i) * nthread;
    }

    const double shift = -0.5 * static_cast<double>(ngrid - 1);
    const unsigned int fftShift = (ngrid + 1) / 2;
    const GComplex *slice = fftField + islice * ngrid2;
    double fpower = 0.0;
    double fx1 = 0.0;
    double fx2 = 0.0;
    double fy1 = 0.0;
    double fy2 = 0.0;

    for (std::size_t logical = static_cast<std::size_t>(tid); logical < ngrid2; logical += nthread) {
        const unsigned int iy = static_cast<unsigned int>(logical / ngrid);
        const unsigned int ix = static_cast<unsigned int>(logical - static_cast<std::size_t>(iy) * ngrid);
        const double dx = static_cast<double>(ix) + shift;
        const double dy = static_cast<double>(iy) + shift;
        const unsigned int iiy = (iy + fftShift) % ngrid;
        const unsigned int iix = (ix + fftShift) % ngrid;
        const std::size_t ii = static_cast<std::size_t>(iiy) * ngrid + iix;
        const GComplex loc = slice[ii];
        const double wei = loc.x * loc.x + loc.y * loc.y;
        fpower += wei;
        fx1 += dx * wei;
        fx2 += dx * dx * wei;
        fy1 += dy * wei;
        fy2 += dy * dy * wei;
    }

    v[0][tid] = fpower;
    v[1][tid] = fx1;
    v[2][tid] = fx2;
    v[3][tid] = fy1;
    v[4][tid] = fy2;
    __syncthreads();
    for (int stride = nthread / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int i = 0; i < NV; ++i) {
                v[i][tid] += v[i][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[islice].fpower = v[0][0];
        out[islice].fx1 = v[1][0];
        out[islice].fx2 = v[2][0];
        out[islice].fy1 = v[3][0];
        out[islice].fy2 = v[4][0];
    }
}

__global__ void buildRImplicitXKernel(const GComplex *field,
                                      const GComplex *source,
                                      GComplex *r,
                                      unsigned int ngrid,
                                      GComplex cstep) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    if (idx >= n) {
        return;
    }

    const unsigned int row = static_cast<unsigned int>(idx / ngrid);
    GComplex lap = scale(field[idx], -2.0);
    if (row > 0) {
        lap = add(lap, field[idx - ngrid]);
    }
    if (row + 1 < ngrid) {
        lap = add(lap, field[idx + ngrid]);
    }

    r[idx] = add(add(source[idx], field[idx]), mul(cstep, lap));
}

__global__ void buildRImplicitYKernel(const GComplex *field,
                                      const GComplex *source,
                                      GComplex *r,
                                      unsigned int ngrid,
                                      GComplex cstep) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    if (idx >= n) {
        return;
    }

    const unsigned int col = static_cast<unsigned int>(idx % ngrid);
    GComplex lap = scale(field[idx], -2.0);
    if (col > 0) {
        lap = add(lap, field[idx - 1]);
    }
    if (col + 1 < ngrid) {
        lap = add(lap, field[idx + 1]);
    }

    r[idx] = add(add(source[idx], field[idx]), mul(cstep, lap));
}

__global__ void tridagXKernel(GComplex *field,
                              const GComplex *r,
                              const GComplex *c,
                              const GComplex *cbet,
                              const GComplex *cwet,
                              unsigned int ngrid) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ngrid) {
        return;
    }

    const unsigned int base = row * ngrid;
    field[base] = mul(r[base], cbet[0]);
    for (unsigned int k = 1; k < ngrid; ++k) {
        const unsigned int idx = base + k;
        field[idx] = mul(sub(r[idx], mul(c[k], field[idx - 1])), cbet[k]);
    }
    for (int k = static_cast<int>(ngrid) - 2; k >= 0; --k) {
        const unsigned int idx = base + static_cast<unsigned int>(k);
        field[idx] = sub(field[idx], mul(cwet[k + 1], field[idx + 1]));
    }
}

__global__ void tridagYKernel(GComplex *field,
                              const GComplex *r,
                              const GComplex *c,
                              const GComplex *cbet,
                              const GComplex *cwet,
                              unsigned int ngrid) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ngrid) {
        return;
    }

    field[col] = mul(r[col], cbet[0]);
    for (unsigned int k = 1; k < ngrid; ++k) {
        const unsigned int idx = k * ngrid + col;
        field[idx] = mul(sub(r[idx], mul(c[k], field[idx - ngrid])), cbet[k]);
    }
    for (int k = static_cast<int>(ngrid) - 2; k >= 0; --k) {
        const unsigned int idx = static_cast<unsigned int>(k) * ngrid + col;
        field[idx] = sub(field[idx], mul(cwet[k + 1], field[idx + ngrid]));
    }
}

inline bool ensureComplexBuffer(GComplex **ptr,
                                std::size_t *capacity,
                                std::size_t required,
                                const char *name) {
    if (*capacity >= required) {
        return true;
    }
    trackedCudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (required == 0) {
        return true;
    }
    if (!setError(name, trackedCudaMalloc(reinterpret_cast<void **>(ptr), required * sizeof(GComplex), name))) {
        return false;
    }
    *capacity = required;
    return true;
}

inline bool ensureParticleBuffer(Particle **ptr,
                                 std::size_t *capacity,
                                 std::size_t required,
                                 const char *name) {
    if (*capacity >= required) {
        return true;
    }
    trackedCudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (required == 0) {
        return true;
    }
    if (!setError(name, trackedCudaMalloc(reinterpret_cast<void **>(ptr), required * sizeof(Particle), name))) {
        return false;
    }
    *capacity = required;
    return true;
}

inline void destroyFFTPlan(cufftHandle *plan, bool *ready, std::size_t *workspaceEstimate = nullptr) {
    if (*ready) {
        cufftDestroy(*plan);
        if (workspaceEstimate != nullptr) {
            recordCufftPlanDestroy(*workspaceEstimate);
            *workspaceEstimate = 0;
        }
        *ready = false;
        *plan = 0;
    }
}


} // namespace

struct State {
    unsigned int ngrid {0};
    std::size_t maxParticles {0};
    std::size_t fieldCapacity {0};
    std::size_t sourceCapacity {0};
    std::size_t rCapacity {0};
    std::size_t coeffCapacity {0};
    std::size_t fftPropagatorCapacity {0};
    std::size_t fftSliceScaleCapacity {0};
    std::size_t fftBatch {0};
    GComplex *field {nullptr};
    GComplex *source {nullptr};
    GComplex *r {nullptr};
    GComplex *c {nullptr};
    GComplex *cbet {nullptr};
    GComplex *cwet {nullptr};
    GComplex *fftK2 {nullptr};
    GComplex *fftSigmoid {nullptr};
    GComplex *mpiSlice {nullptr};
    std::size_t mpiSliceCapacity {0};
    double *fftSliceScale {nullptr};
    Particle *particles {nullptr};
    FieldSliceDiagnostic *diagField {nullptr};
    std::size_t diagFieldCapacity {0};
    cufftHandle fftPlan {0};
    bool fftPlanReady {false};
    std::size_t fftWorkspaceEstimateBytes {0};
};

namespace {

inline void resetGridDependentBuffers(State *state) {
    if (state == nullptr) {
        return;
    }
    destroyFFTPlan(&state->fftPlan, &state->fftPlanReady, &state->fftWorkspaceEstimateBytes);
    trackedCudaFree(state->field);      state->field = nullptr;      state->fieldCapacity = 0;
    trackedCudaFree(state->source);     state->source = nullptr;     state->sourceCapacity = 0;
    trackedCudaFree(state->r);          state->r = nullptr;          state->rCapacity = 0;
    trackedCudaFree(state->c);          state->c = nullptr;
    trackedCudaFree(state->cbet);       state->cbet = nullptr;
    trackedCudaFree(state->cwet);       state->cwet = nullptr;       state->coeffCapacity = 0;
    trackedCudaFree(state->fftK2);      state->fftK2 = nullptr;
    trackedCudaFree(state->fftSigmoid); state->fftSigmoid = nullptr; state->fftPropagatorCapacity = 0;
    trackedCudaFree(state->mpiSlice); state->mpiSlice = nullptr; state->mpiSliceCapacity = 0;
    trackedCudaFree(state->fftSliceScale); state->fftSliceScale = nullptr; state->fftSliceScaleCapacity = 0;
    trackedCudaFree(state->diagField); state->diagField = nullptr; state->diagFieldCapacity = 0;
    state->fftBatch = 0;
}

inline bool validateGridAndBatch(const State *state, unsigned int ngrid, std::size_t batchSize, const char *context) {
    if ((state == nullptr) || (state->ngrid != ngrid) || (batchSize == 0)) {
        g_lastError = std::string(context) + " called before CUDA FFT buffers were initialized";
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = n * batchSize;
    if ((state->fieldCapacity < total) || (state->sourceCapacity < total) || (state->fftBatch < batchSize)) {
        g_lastError = std::string(context) + " called with insufficient CUDA FFT buffer capacity";
        return false;
    }
    return true;
}

inline bool launchInPlaceSlippage(State *state,
                                  const GComplex *boundary,
                                  unsigned int ngrid,
                                  std::size_t batchSize,
                                  int direction,
                                  bool zeroBoundary,
                                  const char *kernelName) {
    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * ngrid;
    const int block = 256;
    const int grid = static_cast<int>((ngrid2 + block - 1) / block);
    fftFieldSlippageInPlaceKernel<<<grid, block>>>(state->field,
                                                   zeroBoundary ? nullptr : boundary,
                                                   ngrid2,
                                                   batchSize,
                                                   direction,
                                                   zeroBoundary ? 1 : 0);
    return checkKernel(kernelName);
}

} // namespace

bool hasDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        setError("cudaGetDeviceCount", err);
        return false;
    }
    if (count < 1) {
        g_lastError = "no CUDA device detected";
        return false;
    }
    return true;
}

const char *lastError() {
    return g_lastError.empty() ? "unknown CUDA error" : g_lastError.c_str();
}

State *create() {
    return new State();
}

void destroy(State *state) {
    if (state == nullptr) {
        return;
    }
    destroyFFTPlan(&state->fftPlan, &state->fftPlanReady, &state->fftWorkspaceEstimateBytes);
    trackedCudaFree(state->field);
    trackedCudaFree(state->source);
    trackedCudaFree(state->r);
    trackedCudaFree(state->c);
    trackedCudaFree(state->cbet);
    trackedCudaFree(state->cwet);
    trackedCudaFree(state->fftK2);
    trackedCudaFree(state->fftSigmoid);
    trackedCudaFree(state->mpiSlice);
    trackedCudaFree(state->fftSliceScale);
    trackedCudaFree(state->particles);
    trackedCudaFree(state->diagField);
    delete state;
}

bool ensure(State *state, unsigned int ngrid, std::size_t maxParticles) {
    if (state == nullptr) {
        g_lastError = "null CUDA state";
        return false;
    }

    if (state->ngrid != ngrid) {
        resetGridDependentBuffers(state);
        state->ngrid = ngrid;
    }

    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    if (!ensureComplexBuffer(&state->field, &state->fieldCapacity, n, "cudaMalloc(field)")) return false;
    if (!ensureComplexBuffer(&state->source, &state->sourceCapacity, n, "cudaMalloc(source)")) return false;
    if (!ensureComplexBuffer(&state->r, &state->rCapacity, n, "cudaMalloc(r)")) return false;
    if (!ensureComplexBuffer(&state->c, &state->coeffCapacity, ngrid, "cudaMalloc(c)")) return false;

    // c/cbet/cwet are three independent arrays.  coeffCapacity tracks their common length.
    if (state->coeffCapacity < ngrid) {
        g_lastError = "internal coefficient buffer capacity mismatch";
        return false;
    }
    if (state->cbet == nullptr) {
        if (!setError("cudaMalloc(cbet)", trackedCudaMalloc(reinterpret_cast<void **>(&state->cbet), ngrid * sizeof(GComplex), "cudaMalloc(cbet)"))) return false;
    }
    if (state->cwet == nullptr) {
        if (!setError("cudaMalloc(cwet)", trackedCudaMalloc(reinterpret_cast<void **>(&state->cwet), ngrid * sizeof(GComplex), "cudaMalloc(cwet)"))) return false;
    }

    if (!ensureParticleBuffer(&state->particles, &state->maxParticles, maxParticles, "cudaMalloc(particles)")) return false;
    return true;
}

bool uploadCoefficients(State *state,
                        const std::complex<double> *c,
                        const std::complex<double> *cbet,
                        const std::complex<double> *cwet,
                        unsigned int ngrid) {
    if ((state == nullptr) || (state->ngrid != ngrid)) {
        g_lastError = "CUDA coefficient upload called before device buffers were initialized";
        return false;
    }
    if (!setError("cudaMemcpy(c)", cudaMemcpy(state->c, reinterpret_cast<const GComplex *>(c), ngrid * sizeof(GComplex), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(cbet)", cudaMemcpy(state->cbet, reinterpret_cast<const GComplex *>(cbet), ngrid * sizeof(GComplex), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(cwet)", cudaMemcpy(state->cwet, reinterpret_cast<const GComplex *>(cwet), ngrid * sizeof(GComplex), cudaMemcpyHostToDevice))) return false;
    return true;
}

bool clearSource(State *state, unsigned int ngrid) {
    if ((state == nullptr) || (state->ngrid != ngrid)) {
        g_lastError = "CUDA source clear called before device buffers were initialized";
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    clearComplexKernel<<<grid, block>>>(state->source, n);
    return checkKernel("clear source kernel");
}

bool buildSource(State *state,
                 const Particle *particles,
                 std::size_t npart,
                 unsigned int ngrid,
                 double gridmax,
                 double dgrid,
                 int harm,
                 double scaleFactor,
                 double undAx,
                 double undAy,
                 double undKx,
                 double undKy,
                 double undGradx,
                 double undGrady) {
    return buildFFTSourceAt(state,
                            particles,
                            npart,
                            ngrid,
                            0,
                            gridmax,
                            dgrid,
                            harm,
                            scaleFactor,
                            undAx,
                            undAy,
                            undKx,
                            undKy,
                            undGradx,
                            undGrady);
}

bool adiStep(State *state,
             std::complex<double> *field,
             unsigned int ngrid,
             std::complex<double> cstepStd) {
    if ((state == nullptr) || (state->ngrid != ngrid)) {
        g_lastError = "CUDA ADI step called before device buffers were initialized";
        return false;
    }

    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    if (!setError("cudaMemcpy(field H2D)", cudaMemcpy(state->field, reinterpret_cast<const GComplex *>(field), n * sizeof(GComplex), cudaMemcpyHostToDevice))) return false;

    const GComplex cstep = makeComplex(cstepStd.real(), cstepStd.imag());
    const int block = 256;
    const int gridAll = static_cast<int>((n + block - 1) / block);
    const int gridLines = static_cast<int>((ngrid + block - 1) / block);

    buildRImplicitXKernel<<<gridAll, block>>>(state->field, state->source, state->r, ngrid, cstep);
    if (!checkKernel("build implicit-x RHS kernel")) return false;

    tridagXKernel<<<gridLines, block>>>(state->field, state->r, state->c, state->cbet, state->cwet, ngrid);
    if (!checkKernel("tridiagonal x kernel")) return false;

    buildRImplicitYKernel<<<gridAll, block>>>(state->field, state->source, state->r, ngrid, cstep);
    if (!checkKernel("build implicit-y RHS kernel")) return false;

    tridagYKernel<<<gridLines, block>>>(state->field, state->r, state->c, state->cbet, state->cwet, ngrid);
    if (!checkKernel("tridiagonal y kernel")) return false;

    if (!setError("cudaMemcpy(field D2H)", cudaMemcpy(reinterpret_cast<GComplex *>(field), state->field, n * sizeof(GComplex), cudaMemcpyDeviceToHost))) return false;
    return true;
}

bool ensureFFT(State *state, unsigned int ngrid, std::size_t maxParticles, std::size_t batchSize) {
    if (state == nullptr) {
        g_lastError = "null CUDA state";
        return false;
    }
    if (batchSize == 0) {
        g_lastError = "CUDA FFT batch size must be positive";
        return false;
    }
    if ((ngrid > static_cast<unsigned int>(INT_MAX)) || (batchSize > static_cast<std::size_t>(INT_MAX))) {
        g_lastError = "CUDA FFT grid or batch size exceeds cuFFT int limits";
        return false;
    }

    if (state->ngrid != ngrid) {
        resetGridDependentBuffers(state);
        state->ngrid = ngrid;
    }

    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = n * batchSize;
    if (!ensureComplexBuffer(&state->field, &state->fieldCapacity, total, "cudaMalloc(fft field)")) return false;
    if (!ensureComplexBuffer(&state->source, &state->sourceCapacity, total, "cudaMalloc(fft source)")) return false;
    if (!ensureComplexBuffer(&state->fftK2, &state->fftPropagatorCapacity, n, "cudaMalloc(fft K2)")) return false;
    if (state->fftSigmoid == nullptr) {
        if (!setError("cudaMalloc(fft sigmoid)", trackedCudaMalloc(reinterpret_cast<void **>(&state->fftSigmoid), n * sizeof(GComplex), "cudaMalloc(fft sigmoid)"))) return false;
    }
    if (!lazyParticleStagingEnabled()) {
        if (!ensureParticleBuffer(&state->particles, &state->maxParticles, maxParticles, "cudaMalloc(particles)")) return false;
    }

    if ((!state->fftPlanReady) || (state->fftBatch != batchSize)) {
        destroyFFTPlan(&state->fftPlan, &state->fftPlanReady, &state->fftWorkspaceEstimateBytes);
        int dims[2] = {static_cast<int>(ngrid), static_cast<int>(ngrid)};
        int embed[2] = {static_cast<int>(ngrid), static_cast<int>(ngrid)};
        const int stride = 1;
        const int dist = static_cast<int>(n);
        std::size_t estimatedWorkspace = 0;
        cufftEstimateMany(2,
                          dims,
                          embed,
                          stride,
                          dist,
                          embed,
                          stride,
                          dist,
                          CUFFT_Z2Z,
                          static_cast<int>(batchSize),
                          &estimatedWorkspace);
        if (!setCufftError("cufftPlanMany(FFT field solver)",
                           cufftPlanMany(&state->fftPlan,
                                         2,
                                         dims,
                                         embed,
                                         stride,
                                         dist,
                                         embed,
                                         stride,
                                         dist,
                                         CUFFT_Z2Z,
                                         static_cast<int>(batchSize)))) {
            return false;
        }
        state->fftPlanReady = true;
        state->fftWorkspaceEstimateBytes = estimatedWorkspace;
        recordCufftPlanCreate(estimatedWorkspace);
        state->fftBatch = batchSize;
    }

    return true;
}

bool uploadFFTPropagator(State *state,
                         const std::complex<double> *K2,
                         const std::complex<double> *sigmoid,
                         unsigned int ngrid) {
    if ((state == nullptr) || (state->ngrid != ngrid) || (state->fftK2 == nullptr) || (state->fftSigmoid == nullptr)) {
        g_lastError = "CUDA FFT propagator upload called before buffers were initialized";
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    if (!setError("cudaMemcpy(fft K2)", cudaMemcpy(state->fftK2, reinterpret_cast<const GComplex *>(K2), n * sizeof(GComplex), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(fft sigmoid)", cudaMemcpy(state->fftSigmoid, reinterpret_cast<const GComplex *>(sigmoid), n * sizeof(GComplex), cudaMemcpyHostToDevice))) return false;
    return true;
}

bool uploadFFTFields(State *state,
                     const std::complex<double> *fields,
                     unsigned int ngrid,
                     std::size_t batchSize) {
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA FFT field upload")) {
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = n * batchSize;
    return setError("cudaMemcpy(fft fields H2D)", cudaMemcpy(state->field, reinterpret_cast<const GComplex *>(fields), total * sizeof(GComplex), cudaMemcpyHostToDevice));
}

bool downloadFFTFields(State *state,
                       std::complex<double> *fields,
                       unsigned int ngrid,
                       std::size_t batchSize) {
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA FFT field download")) {
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = n * batchSize;
    return setError("cudaMemcpy(fft fields D2H)", cudaMemcpy(reinterpret_cast<GComplex *>(fields), state->field, total * sizeof(GComplex), cudaMemcpyDeviceToHost));
}

bool fftFieldApplySlippage(State *state,
                           unsigned int ngrid,
                           std::size_t batchSize,
                           int direction) {
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA FFT field slippage")) {
        return false;
    }
    if ((direction != 1) && (direction != -1)) {
        g_lastError = "CUDA FFT field slippage direction must be +/-1";
        return false;
    }

    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = ngrid2 * batchSize;

    if (inplaceSlippageEnabled()) {
        return launchInPlaceSlippage(state,
                                     nullptr,
                                     ngrid,
                                     batchSize,
                                     direction,
                                     true,
                                     "FFT field in-place slippage kernel");
    }

    if (state->sourceCapacity < total) {
        g_lastError = "CUDA FFT field slippage requires a source/scratch buffer matching the field size";
        return false;
    }

    if (!setError("cudaMemcpy(FFT field slippage scratch D2D)",
                  cudaMemcpy(state->source,
                             state->field,
                             total * sizeof(GComplex),
                             cudaMemcpyDeviceToDevice))) {
        return false;
    }

    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    fftFieldSlippageKernel<<<grid, block>>>(state->source,
                                            state->field,
                                            ngrid2,
                                            batchSize,
                                            direction);
    return checkKernel("FFT field slippage kernel");
}

bool fftFieldDownloadSlippageSlice(State *state,
                                   double *hostBuffer,
                                   std::size_t hostDoubles,
                                   unsigned int ngrid,
                                   std::size_t batchSize,
                                   int direction) {
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA FFT slippage boundary download")) {
        return false;
    }
    if ((hostBuffer == nullptr) || ((direction != 1) && (direction != -1))) {
        g_lastError = "CUDA FFT slippage boundary download called with invalid arguments";
        return false;
    }

    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * ngrid;
    if (hostDoubles < 2 * ngrid2) {
        g_lastError = "CUDA FFT slippage boundary host buffer is too small";
        return false;
    }

    const std::size_t sendSlice = (direction > 0) ? (batchSize - 1) : 0;
    return setError("cudaMemcpy(FFT slippage boundary D2H)",
                    cudaMemcpy(reinterpret_cast<GComplex *>(hostBuffer),
                               state->field + sendSlice * ngrid2,
                               ngrid2 * sizeof(GComplex),
                               cudaMemcpyDeviceToHost));
}

bool fftFieldApplySlippageBoundary(State *state,
                                   const double *hostBoundary,
                                   std::size_t hostDoubles,
                                   unsigned int ngrid,
                                   std::size_t batchSize,
                                   int direction,
                                   bool zeroBoundary) {
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA FFT slippage boundary apply")) {
        return false;
    }
    if ((direction != 1) && (direction != -1)) {
        g_lastError = "CUDA FFT slippage boundary direction must be +/-1";
        return false;
    }

    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = ngrid2 * batchSize;

    if (!zeroBoundary) {
        if ((hostBoundary == nullptr) || (hostDoubles < 2 * ngrid2)) {
            g_lastError = "CUDA FFT slippage boundary host buffer is invalid";
            return false;
        }
        if (!ensureComplexBuffer(&state->mpiSlice,
                                 &state->mpiSliceCapacity,
                                 ngrid2,
                                 "cudaMalloc(FFT slippage MPI slice)")) {
            return false;
        }
        if (!setError("cudaMemcpy(FFT slippage boundary H2D)",
                      cudaMemcpy(state->mpiSlice,
                                 reinterpret_cast<const GComplex *>(hostBoundary),
                                 ngrid2 * sizeof(GComplex),
                                 cudaMemcpyHostToDevice))) {
            return false;
        }
    } else if (!inplaceSlippageEnabled() && state->mpiSlice == nullptr) {
        // The legacy scratch kernel does not dereference the boundary pointer
        // when zeroBoundary is set, but pass a non-null address for defensive
        // compatibility with older CUDA debug tooling.
        if (!ensureComplexBuffer(&state->mpiSlice,
                                 &state->mpiSliceCapacity,
                                 1,
                                 "cudaMalloc(FFT slippage zero boundary sentinel)")) {
            return false;
        }
    }

    if (inplaceSlippageEnabled()) {
        return launchInPlaceSlippage(state,
                                     state->mpiSlice,
                                     ngrid,
                                     batchSize,
                                     direction,
                                     zeroBoundary,
                                     "FFT field in-place MPI slippage boundary kernel");
    }

    if (state->sourceCapacity < total) {
        g_lastError = "CUDA FFT slippage boundary requires a source/scratch buffer matching the field size";
        return false;
    }

    if (!setError("cudaMemcpy(FFT slippage boundary scratch D2D)",
                  cudaMemcpy(state->source,
                             state->field,
                             total * sizeof(GComplex),
                             cudaMemcpyDeviceToDevice))) {
        return false;
    }

    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    fftFieldSlippageBoundaryKernel<<<grid, block>>>(state->source,
                                                    state->mpiSlice,
                                                    state->field,
                                                    ngrid2,
                                                    batchSize,
                                                    direction,
                                                    zeroBoundary ? 1 : 0);
    return checkKernel("FFT field MPI slippage boundary kernel");
}

bool clearFFTSource(State *state, unsigned int ngrid, std::size_t batchSize) {
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA FFT source clear")) {
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = n * batchSize;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    clearComplexKernel<<<grid, block>>>(state->source, total);
    return checkKernel("clear FFT source kernel");
}

bool buildFFTSourceAt(State *state,
                      const Particle *particles,
                      std::size_t npart,
                      unsigned int ngrid,
                      std::size_t sourceOffset,
                      double gridmax,
                      double dgrid,
                      int harm,
                      double scaleFactor,
                      double undAx,
                      double undAy,
                      double undKx,
                      double undKy,
                      double undGradx,
                      double undGrady) {
    if ((state == nullptr) || (state->ngrid != ngrid)) {
        g_lastError = "CUDA source build called before device buffers were initialized";
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    if (sourceOffset + n > state->sourceCapacity) {
        g_lastError = "CUDA source build requested an out-of-range source offset";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    if (state->maxParticles < npart || state->particles == nullptr) {
        if (!ensureParticleBuffer(&state->particles,
                                  &state->maxParticles,
                                  npart,
                                  "cudaMalloc(particles lazy staging)")) {
            return false;
        }
    }

    if (!setError("cudaMemcpy(particles)", cudaMemcpy(state->particles, particles, npart * sizeof(Particle), cudaMemcpyHostToDevice))) return false;

    const int block = sourceDepositionBlockSize();
    const int grid = static_cast<int>((npart + block - 1) / block);
    buildSourceKernel<<<grid, block>>>(state->particles,
                                       npart,
                                       state->source + sourceOffset,
                                       ngrid,
                                       gridmax,
                                       dgrid,
                                       harm,
                                       scaleFactor,
                                       undAx,
                                       undAy,
                                       undKx,
                                       undKy,
                                       undGradx,
                                       undGrady);
    return checkKernel("source deposition kernel");
}

bool executeFFTPropagation(State *state,
                           unsigned int ngrid,
                           std::size_t batchSize,
                           double delz,
                           bool doFilter) {
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA FFT propagation")) {
        return false;
    }
    if (!state->fftPlanReady || (state->fftK2 == nullptr) || (state->fftSigmoid == nullptr)) {
        g_lastError = "CUDA FFT propagation called before cuFFT plan/propagator was initialized";
        return false;
    }

    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = n * batchSize;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);

    if (!setCufftError("cuFFT forward(field)",
                       cufftExecZ2Z(state->fftPlan,
                                    reinterpret_cast<cufftDoubleComplex *>(state->field),
                                    reinterpret_cast<cufftDoubleComplex *>(state->field),
                                    CUFFT_FORWARD))) return false;

    if (!setCufftError("cuFFT forward(source)",
                       cufftExecZ2Z(state->fftPlan,
                                    reinterpret_cast<cufftDoubleComplex *>(state->source),
                                    reinterpret_cast<cufftDoubleComplex *>(state->source),
                                    CUFFT_FORWARD))) return false;

    fftPropagateKernel<<<grid, block>>>(state->field,
                                        state->source,
                                        state->fftK2,
                                        state->fftSigmoid,
                                        n,
                                        total,
                                        delz,
                                        doFilter ? 1 : 0);
    if (!checkKernel("FFT propagation kernel")) return false;

    if (!setCufftError("cuFFT inverse(field)",
                       cufftExecZ2Z(state->fftPlan,
                                    reinterpret_cast<cufftDoubleComplex *>(state->field),
                                    reinterpret_cast<cufftDoubleComplex *>(state->field),
                                    CUFFT_INVERSE))) return false;

    scaleComplexKernel<<<grid, block>>>(state->field, total, 1.0 / static_cast<double>(n));
    return checkKernel("FFT normalization kernel");
}

namespace {

enum TrackMode {
    TRACK_DRIFT = 0,
    TRACK_FOCUS = 1,
    TRACK_DEFOCUS = 2
};

inline bool ensureDoubleBuffer(double **ptr,
                               std::size_t *capacity,
                               std::size_t required,
                               const char *name) {
    if (*capacity >= required) {
        return true;
    }
    trackedCudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (required == 0) {
        return true;
    }
    if (!setError(name, trackedCudaMalloc(reinterpret_cast<void **>(ptr), required * sizeof(double), name))) {
        return false;
    }
    *capacity = required;
    return true;
}

inline bool ensureIntBuffer(int **ptr,
                            std::size_t *capacity,
                            std::size_t required,
                            const char *name) {
    if (*capacity >= required) {
        return true;
    }
    trackedCudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (required == 0) {
        return true;
    }
    if (!setError(name, trackedCudaMalloc(reinterpret_cast<void **>(ptr), required * sizeof(int), name))) {
        return false;
    }
    *capacity = required;
    return true;
}

inline bool ensureSizeBuffer(std::size_t **ptr,
                             std::size_t *capacity,
                             std::size_t required,
                             const char *name) {
    if (*capacity >= required) {
        return true;
    }
    trackedCudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (required == 0) {
        return true;
    }
    if (!setError(name, trackedCudaMalloc(reinterpret_cast<void **>(ptr), required * sizeof(std::size_t), name))) {
        return false;
    }
    *capacity = required;
    return true;
}

inline bool ensureComplexPointerBuffer(GComplex ***ptr,
                                       std::size_t *capacity,
                                       std::size_t required,
                                       const char *name) {
    if (*capacity >= required) {
        return true;
    }
    trackedCudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (required == 0) {
        return true;
    }
    if (!setError(name, trackedCudaMalloc(reinterpret_cast<void **>(ptr), required * sizeof(GComplex *), name))) {
        return false;
    }
    if (!setError("cudaMemset(beam field pointer table)", cudaMemset(*ptr, 0, required * sizeof(GComplex *)))) {
        return false;
    }
    *capacity = required;
    return true;
}

__host__ __device__ inline GComplex conjComplex(GComplex a) {
    return makeComplex(a.x, -a.y);
}

__device__ inline void applyTrackMode(int mode,
                                      double delz,
                                      double qf,
                                      double *coord,
                                      double *mom,
                                      double gammaz,
                                      double offset) {
    if (mode == TRACK_DRIFT) {
        *coord += (*mom) * delz / gammaz;
        return;
    }

    if (mode == TRACK_FOCUS) {
        const double foc = sqrt(qf / gammaz);
        const double omg = foc * delz;
        const double a1 = cos(omg);
        const double a2 = sin(omg) / foc;
        const double a3 = -a2 * foc * foc;
        const double tmp = *coord - offset;
        *coord = a1 * tmp + a2 * (*mom) / gammaz + offset;
        *mom = a3 * tmp * gammaz + a1 * (*mom);
        return;
    }

    const double foc = sqrt(-qf / gammaz);
    const double omg = foc * delz;
    const double a1 = cosh(omg);
    const double a2 = sinh(omg) / foc;
    const double a3 = a2 * foc * foc;
    const double tmp = *coord - offset;
    *coord = a1 * tmp + a2 * (*mom) / gammaz + offset;
    *mom = a3 * tmp * gammaz + a1 * (*mom);
}

__global__ void beamTrackTransverseKernel(double *gamma,
                                          double *x,
                                          double *y,
                                          double *px,
                                          double *py,
                                          std::size_t npart,
                                          double delz,
                                          double aw,
                                          double qx,
                                          double qy,
                                          double xoff,
                                          double yoff,
                                          int modeX,
                                          int modeY) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    const double gammaz = sqrt(gamma[ip] * gamma[ip] - 1.0 - aw * aw - px[ip] * px[ip] - py[ip] * py[ip]);
    double tx = x[ip];
    double tpx = px[ip];
    double ty = y[ip];
    double tpy = py[ip];

    applyTrackMode(modeX, delz, qx, &tx, &tpx, gammaz, xoff);
    applyTrackMode(modeY, delz, qy, &ty, &tpy, gammaz, yoff);

    x[ip] = tx;
    px[ip] = tpx;
    y[ip] = ty;
    py[ip] = tpy;
}

template<int Mode>
__device__ inline void applyTrackModeT(double delz,
                                      double qf,
                                      double *coord,
                                      double *mom,
                                      double gammaz,
                                      double offset) {
    if constexpr (Mode == TRACK_DRIFT) {
        *coord += (*mom) * delz / gammaz;
    } else if constexpr (Mode == TRACK_FOCUS) {
        const double foc = sqrt(qf / gammaz);
        const double omg = foc * delz;
        double somg = 0.0;
        double comg = 1.0;
        sincosPortable(omg, &somg, &comg);
        const double a2 = somg / foc;
        const double a3 = -a2 * foc * foc;
        const double tmp = *coord - offset;
        *coord = comg * tmp + a2 * (*mom) / gammaz + offset;
        *mom = a3 * tmp * gammaz + comg * (*mom);
    } else {
        const double foc = sqrt(-qf / gammaz);
        const double omg = foc * delz;
        const double a1 = cosh(omg);
        const double a2 = sinh(omg) / foc;
        const double a3 = a2 * foc * foc;
        const double tmp = *coord - offset;
        *coord = a1 * tmp + a2 * (*mom) / gammaz + offset;
        *mom = a3 * tmp * gammaz + a1 * (*mom);
    }
}

template<int ModeX, int ModeY>
__global__ void beamTrackTransverseKernelT(double *gamma,
                                           double *x,
                                           double *y,
                                           double *px,
                                           double *py,
                                           std::size_t npart,
                                           double delz,
                                           double aw,
                                           double qx,
                                           double qy,
                                           double xoff,
                                           double yoff) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    const double tgamma = gamma[ip];
    const double tpx0 = px[ip];
    const double tpy0 = py[ip];
    const double gammaz = sqrt(tgamma * tgamma - 1.0 - aw * aw - tpx0 * tpx0 - tpy0 * tpy0);
    double tx = x[ip];
    double tpx = tpx0;
    double ty = y[ip];
    double tpy = tpy0;

    applyTrackModeT<ModeX>(delz, qx, &tx, &tpx, gammaz, xoff);
    applyTrackModeT<ModeY>(delz, qy, &ty, &tpy, gammaz, yoff);

    x[ip] = tx;
    px[ip] = tpx;
    y[ip] = ty;
    py[ip] = tpy;
}


__global__ void beamTrackTransverseFocusSymmetricKernel(double *gamma,
                                                       double *x,
                                                       double *y,
                                                       double *px,
                                                       double *py,
                                                       std::size_t npart,
                                                       double delz,
                                                       double aw,
                                                       double qf,
                                                       double xoff,
                                                       double yoff) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    const double tgamma = gamma[ip];
    const double tpx0 = px[ip];
    const double tpy0 = py[ip];
    const double gammaz = sqrt(tgamma * tgamma - 1.0 - aw * aw - tpx0 * tpx0 - tpy0 * tpy0);

    // Helical undulator natural focusing often produces modeX=modeY=focus with
    // qx == qy.  In that common case, x and y share the same transfer matrix.
    // Compute sqrt/sincos once per particle instead of once per plane.
    const double foc = sqrt(qf / gammaz);
    const double omg = foc * delz;
    double somg = 0.0;
    double comg = 1.0;
    sincosPortable(omg, &somg, &comg);
    const double a2 = somg / foc;
    const double a3 = -a2 * foc * foc;

    const double tx0 = x[ip];
    const double ty0 = y[ip];
    const double dx = tx0 - xoff;
    const double dy = ty0 - yoff;

    x[ip]  = comg * dx + a2 * tpx0 / gammaz + xoff;
    px[ip] = a3 * dx * gammaz + comg * tpx0;
    y[ip]  = comg * dy + a2 * tpy0 / gammaz + yoff;
    py[ip] = a3 * dy * gammaz + comg * tpy0;
}

__global__ void beamCorrectorKernel(double *px,
                                    double *py,
                                    std::size_t npart,
                                    double cx,
                                    double cy) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip < npart) {
        px[ip] += cx;
        py[ip] += cy;
    }
}

__global__ void beamR56Kernel(double *gamma,
                              double *theta,
                              std::size_t npart,
                              double r56,
                              double gamma0) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip < npart) {
        theta[ip] += r56 * (gamma[ip] - gamma0);
    }
}

__global__ void beamChicaneMatrixKernel(double *gamma,
                                        double *x,
                                        double *y,
                                        double *px,
                                        double *py,
                                        std::size_t npart,
                                        const double *m) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    const double gammaz = sqrt(gamma[ip] * gamma[ip] - 1.0 - px[ip] * px[ip] - py[ip] * py[ip]);
    double tmp = x[ip];
    x[ip] = m[0] * tmp + m[1] * px[ip] / gammaz;
    px[ip] = m[4] * tmp * gammaz + m[5] * px[ip];
    tmp = y[ip];
    y[ip] = m[10] * tmp + m[11] * py[ip] / gammaz;
    py[ip] = m[14] * tmp * gammaz + m[15] * py[ip];
}

__device__ inline double undulatorAwLocal(double x,
                                          double y,
                                          double undAx,
                                          double undAy,
                                          double undKx,
                                          double undKy,
                                          double undGradx,
                                          double undGrady) {
    const double dx = x - undAx;
    const double dy = y - undAy;
    return 1.0 + 0.5 * (undKx * dx * dx + undKy * dy * dy) + undGradx * dx + undGrady * dy;
}

__device__ inline bool loadInterpolatedField(const GComplex *fields,
                                             std::size_t base,
                                             int ngrid,
                                             double gridmax,
                                             double dgrid,
                                             double x,
                                             double y,
                                             GComplex *value) {
    if (!((x > -gridmax) && (x < gridmax) && (y > -gridmax) && (y < gridmax))) {
        *value = makeComplex(0.0, 0.0);
        return false;
    }

    const double wx0 = (x + gridmax) / dgrid;
    const double wy0 = (y + gridmax) / dgrid;
    const int ix = static_cast<int>(floor(wx0));
    const int iy = static_cast<int>(floor(wy0));
    const double wx = 1.0 + floor(wx0) - wx0;
    const double wy = 1.0 + floor(wy0) - wy0;
    int idx = ix + iy * ngrid;

    GComplex acc = scale(fields[base + idx], wx * wy);
    idx += 1;
    acc = add(acc, scale(fields[base + idx], (1.0 - wx) * wy));
    idx += ngrid - 1;
    acc = add(acc, scale(fields[base + idx], wx * (1.0 - wy)));
    idx += 1;
    acc = add(acc, scale(fields[base + idx], (1.0 - wx) * (1.0 - wy)));
    *value = acc;
    return true;
}

__device__ inline GComplex longitudinalCtmp(const double *x,
                                            const double *y,
                                            const GComplex *fields,
                                            GComplex * const *fieldData,
                                            const std::size_t *fieldOffset,
                                            const int *fieldNgrid,
                                            const int *fieldNslices,
                                            const int *fieldHarm,
                                            const double *fieldGridmax,
                                            const double *fieldDgrid,
                                            const double *fieldRtmp,
                                            std::size_t nfield,
                                            std::size_t ip,
                                            int slice,
                                            double awloc,
                                            double ttheta) {
    GComplex ctmp = makeComplex(0.0, 0.0);
    const double px = x[ip];
    const double py = y[ip];

    for (std::size_t ifld = 0; ifld < nfield; ++ifld) {
        const int ngrid = fieldNgrid[ifld];
        const int nslices = fieldNslices[ifld];
        if ((ngrid < 2) || (nslices < 1)) {
            continue;
        }
        const int fslice = ((slice % nslices) + nslices) % nslices;
        const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * static_cast<std::size_t>(ngrid);

        const GComplex *fieldBase = nullptr;
        if (fieldData != nullptr) {
            fieldBase = fieldData[ifld];
        }
        std::size_t base = static_cast<std::size_t>(fslice) * ngrid2;
        if (fieldBase == nullptr) {
            if (fields == nullptr) {
                continue;
            }
            fieldBase = fields;
            base += fieldOffset[ifld];
        }

        GComplex cpart;
        if (!loadInterpolatedField(fieldBase,
                                   base,
                                   ngrid,
                                   fieldGridmax[ifld],
                                   fieldDgrid[ifld],
                                   px,
                                   py,
                                   &cpart)) {
            continue;
        }

        GComplex rpart = scale(conjComplex(cpart), fieldRtmp[ifld] * awloc);
        const double phase = static_cast<double>(fieldHarm[ifld]) * ttheta;
        double sphase = 0.0;
        double cphase = 1.0;
        sincosPortable(phase, &sphase, &cphase);
        const GComplex expPhase = makeComplex(cphase, -sphase);
        ctmp = add(ctmp, mul(rpart, expPhase));
    }
    return ctmp;
}

__device__ inline GComplex longitudinalCtmpOneField(double px,
                                                    double py,
                                                    const GComplex *fieldBase,
                                                    int ngrid,
                                                    int nslices,
                                                    int harm,
                                                    double gridmax,
                                                    double dgrid,
                                                    double rtmp,
                                                    int slice,
                                                    double awloc,
                                                    double ttheta) {
    if ((fieldBase == nullptr) || (ngrid < 2) || (nslices < 1)) {
        return makeComplex(0.0, 0.0);
    }

    const int fslice = ((slice % nslices) + nslices) % nslices;
    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * static_cast<std::size_t>(ngrid);
    const std::size_t base = static_cast<std::size_t>(fslice) * ngrid2;

    GComplex cpart;
    if (!loadInterpolatedField(fieldBase,
                               base,
                               ngrid,
                               gridmax,
                               dgrid,
                               px,
                               py,
                               &cpart)) {
        return makeComplex(0.0, 0.0);
    }

    GComplex rpart = scale(conjComplex(cpart), rtmp * awloc);
    const double phase = static_cast<double>(harm) * ttheta;
    double sphase = 0.0;
    double cphase = 1.0;
    sincosPortable(phase, &sphase, &cphase);
    const GComplex expPhase = makeComplex(cphase, -sphase);
    return mul(rpart, expPhase);
}

__device__ inline GComplex longitudinalAmplitudeOneField(double px,
                                                         double py,
                                                         const GComplex *fieldBase,
                                                         int ngrid,
                                                         int nslices,
                                                         double gridmax,
                                                         double dgrid,
                                                         double rtmp,
                                                         int slice,
                                                         double awloc) {
    if ((fieldBase == nullptr) || (ngrid < 2) || (nslices < 1)) {
        return makeComplex(0.0, 0.0);
    }

    const int fslice = ((slice % nslices) + nslices) % nslices;
    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * static_cast<std::size_t>(ngrid);
    const std::size_t base = static_cast<std::size_t>(fslice) * ngrid2;

    GComplex cpart;
    if (!loadInterpolatedField(fieldBase,
                               base,
                               ngrid,
                               gridmax,
                               dgrid,
                               px,
                               py,
                               &cpart)) {
        return makeComplex(0.0, 0.0);
    }

    return scale(conjComplex(cpart), rtmp * awloc);
}

__device__ inline GComplex longitudinalCtmpFromAmplitude(GComplex rpart,
                                                         double harm,
                                                         double ttheta) {
    const double phase = harm * ttheta;
    double sphase = 0.0;
    double cphase = 1.0;
    sincosPortable(phase, &sphase, &cphase);
    const GComplex expPhase = makeComplex(cphase, -sphase);
    return mul(rpart, expPhase);
}

__device__ inline void longitudinalOdeAddOneField(double px,
                                                  double py,
                                                  const GComplex *fieldBase,
                                                  int ngrid,
                                                  int nslices,
                                                  int harm,
                                                  double gridmax,
                                                  double dgrid,
                                                  double rtmp,
                                                  int slice,
                                                  double awloc,
                                                  double btpar,
                                                  double xks,
                                                  double xku,
                                                  double ez,
                                                  double tgam,
                                                  double tthet,
                                                  double *kgg,
                                                  double *kpp) {
    const GComplex ctmp = longitudinalCtmpOneField(px,
                                                   py,
                                                   fieldBase,
                                                   ngrid,
                                                   nslices,
                                                   harm,
                                                   gridmax,
                                                   dgrid,
                                                   rtmp,
                                                   slice,
                                                   awloc,
                                                   tthet);
    const double ztemp1 = -2.0 / xks;
    const double btper0 = btpar + ztemp1 * ctmp.x;
    const double btpar0 = sqrt(1.0 - btper0 / (tgam * tgam));
    *kpp += xks * (1.0 - 1.0 / btpar0) + xku;
    *kgg += ctmp.y / btpar0 / tgam - ez;
}

__device__ inline void longitudinalOdeAddOneFieldCached(GComplex rpart,
                                                        double harm,
                                                        double btpar,
                                                        double ztemp1,
                                                        double xks,
                                                        double xku,
                                                        double ez,
                                                        double tgam,
                                                        double tthet,
                                                        bool algebraOpt,
                                                        double *kgg,
                                                        double *kpp) {
    const GComplex ctmp = longitudinalCtmpFromAmplitude(rpart, harm, tthet);
    const double btper0 = btpar + ztemp1 * ctmp.x;

    if (algebraOpt) {
        // Algebraically equivalent to the original Genesis expression, but arranged
        // to avoid repeated FP64 divisions in the hot RK4 loop:
        //   btpar0 = sqrt(1 - btper0 / (gamma*gamma))
        //   kpp   += xks * (1 - 1/btpar0) + xku
        //   kgg   += Im(ctmp) / btpar0 / gamma - ez
        // ztemp1=-2/xks and harm are invariant over the four RK substeps and are
        // precomputed once per particle in the cached-interpolation kernel.
        const double invGam = 1.0 / tgam;
        const double invBtpar0 = 1.0 / sqrt(1.0 - btper0 * invGam * invGam);
        *kpp += xks * (1.0 - invBtpar0) + xku;
        *kgg += ctmp.y * invBtpar0 * invGam - ez;
        return;
    }

    // Conservative path kept for A/B validation.  It matches the previous
    // Stage 4.2A algebra except that ztemp1 is still precomputed once per
    // particle rather than recomputed during every RK substep.
    const double btpar0 = sqrt(1.0 - btper0 / (tgam * tgam));
    *kpp += xks * (1.0 - 1.0 / btpar0) + xku;
    *kgg += ctmp.y / btpar0 / tgam - ez;
}

__device__ inline void longitudinalOdeAdd(const double *x,
                                          const double *y,
                                          const GComplex *fields,
                                          GComplex * const *fieldData,
                                          const std::size_t *fieldOffset,
                                          const int *fieldNgrid,
                                          const int *fieldNslices,
                                          const int *fieldHarm,
                                          const double *fieldGridmax,
                                          const double *fieldDgrid,
                                          const double *fieldRtmp,
                                          std::size_t nfield,
                                          std::size_t ip,
                                          int slice,
                                          double awloc,
                                          double btpar,
                                          double xks,
                                          double xku,
                                          double ez,
                                          double tgam,
                                          double tthet,
                                          double *kgg,
                                          double *kpp) {
    const GComplex ctmp = longitudinalCtmp(x,
                                           y,
                                           fields,
                                           fieldData,
                                           fieldOffset,
                                           fieldNgrid,
                                           fieldNslices,
                                           fieldHarm,
                                           fieldGridmax,
                                           fieldDgrid,
                                           fieldRtmp,
                                           nfield,
                                           ip,
                                           slice,
                                           awloc,
                                           tthet);
    const double ztemp1 = -2.0 / xks;
    const double btper0 = btpar + ztemp1 * ctmp.x;
    const double btpar0 = sqrt(1.0 - btper0 / (tgam * tgam));
    *kpp += xks * (1.0 - 1.0 / btpar0) + xku;
    *kgg += ctmp.y / btpar0 / tgam - ez;
}

__global__ void beamLongitudinalOneFieldKernel(double *gamma,
                                               double *theta,
                                               const double *x,
                                               const double *y,
                                               const double *px,
                                               const double *py,
                                               const double *ez,
                                               const int *particleSlice,
                                               std::size_t npart,
                                               const GComplex *fieldBase,
                                               int fieldNgrid,
                                               int fieldNslices,
                                               int fieldHarm,
                                               double fieldGridmax,
                                               double fieldDgrid,
                                               double fieldRtmp,
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
                                               double undGrady) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    double tgamma = gamma[ip];
    double ttheta = theta[ip] + autophase;
    const double tx = x[ip];
    const double ty = y[ip];
    const double tpx = px[ip];
    const double tpy = py[ip];
    const double awloc = undulatorAwLocal(tx, ty, undAx, undAy, undKx, undKy, undGradx, undGrady);
    const double btpar = 1.0 + tpx * tpx + tpy * tpy + aw * aw * awloc * awloc;
    const double efield = ez[ip];
    const int slice = particleSlice[ip];

    double k2gg = 0.0;
    double k2pp = 0.0;
    longitudinalOdeAddOneField(tx, ty, fieldBase, fieldNgrid, fieldNslices, fieldHarm,
                               fieldGridmax, fieldDgrid, fieldRtmp, slice, awloc,
                               btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    double stpz = 0.5 * delz;
    tgamma += stpz * k2gg;
    ttheta += stpz * k2pp;

    double k3gg = k2gg;
    double k3pp = k2pp;
    k2gg = 0.0;
    k2pp = 0.0;
    longitudinalOdeAddOneField(tx, ty, fieldBase, fieldNgrid, fieldNslices, fieldHarm,
                               fieldGridmax, fieldDgrid, fieldRtmp, slice, awloc,
                               btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    tgamma += stpz * (k2gg - k3gg);
    ttheta += stpz * (k2pp - k3pp);

    k3gg /= 6.0;
    k3pp /= 6.0;
    k2gg *= -0.5;
    k2pp *= -0.5;
    longitudinalOdeAddOneField(tx, ty, fieldBase, fieldNgrid, fieldNslices, fieldHarm,
                               fieldGridmax, fieldDgrid, fieldRtmp, slice, awloc,
                               btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    stpz = delz;
    tgamma += stpz * k2gg;
    ttheta += stpz * k2pp;

    k3gg -= k2gg;
    k3pp -= k2pp;
    k2gg *= 2.0;
    k2pp *= 2.0;
    longitudinalOdeAddOneField(tx, ty, fieldBase, fieldNgrid, fieldNslices, fieldHarm,
                               fieldGridmax, fieldDgrid, fieldRtmp, slice, awloc,
                               btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    tgamma += stpz * (k3gg + k2gg / 6.0);
    ttheta += stpz * (k3pp + k2pp / 6.0);

    gamma[ip] = tgamma;
    theta[ip] = ttheta;
}

__global__ void beamLongitudinalOneFieldCachedInterpKernel(double *gamma,
                                                           double *theta,
                                                           const double *x,
                                                           const double *y,
                                                           const double *px,
                                                           const double *py,
                                                           const double *ez,
                                                           const int *particleSlice,
                                                           std::size_t npart,
                                                           const GComplex *fieldBase,
                                                           int fieldNgrid,
                                                           int fieldNslices,
                                                           int fieldHarm,
                                                           double fieldGridmax,
                                                           double fieldDgrid,
                                                           double fieldRtmp,
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
                                                           double undGrady,
                                                           bool algebraOpt) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    double tgamma = gamma[ip];
    double ttheta = theta[ip] + autophase;
    const double tx = x[ip];
    const double ty = y[ip];
    const double tpx = px[ip];
    const double tpy = py[ip];
    const double awloc = undulatorAwLocal(tx, ty, undAx, undAy, undKx, undKy, undGradx, undGrady);
    const double btpar = 1.0 + tpx * tpx + tpy * tpy + aw * aw * awloc * awloc;
    const double efield = ez[ip];
    const int slice = particleSlice[ip];

    // x/y/slice are constant during this longitudinal RK4 step.  Cache the
    // bilinear field interpolation once and reuse it for the four RK substeps;
    // only the phase factor depends on the evolving theta.
    const GComplex rpart = longitudinalAmplitudeOneField(tx,
                                                         ty,
                                                         fieldBase,
                                                         fieldNgrid,
                                                         fieldNslices,
                                                         fieldGridmax,
                                                         fieldDgrid,
                                                         fieldRtmp,
                                                         slice,
                                                         awloc);
    const double fieldHarmD = static_cast<double>(fieldHarm);
    const double ztemp1 = -2.0 / xks;

    double k2gg = 0.0;
    double k2pp = 0.0;
    longitudinalOdeAddOneFieldCached(rpart, fieldHarmD, btpar, ztemp1, xks, xku, efield,
                                     tgamma, ttheta, algebraOpt, &k2gg, &k2pp);

    double stpz = 0.5 * delz;
    tgamma += stpz * k2gg;
    ttheta += stpz * k2pp;

    double k3gg = k2gg;
    double k3pp = k2pp;
    k2gg = 0.0;
    k2pp = 0.0;
    longitudinalOdeAddOneFieldCached(rpart, fieldHarmD, btpar, ztemp1, xks, xku, efield,
                                     tgamma, ttheta, algebraOpt, &k2gg, &k2pp);

    tgamma += stpz * (k2gg - k3gg);
    ttheta += stpz * (k2pp - k3pp);

    k3gg /= 6.0;
    k3pp /= 6.0;
    k2gg *= -0.5;
    k2pp *= -0.5;
    longitudinalOdeAddOneFieldCached(rpart, fieldHarmD, btpar, ztemp1, xks, xku, efield,
                                     tgamma, ttheta, algebraOpt, &k2gg, &k2pp);

    stpz = delz;
    tgamma += stpz * k2gg;
    ttheta += stpz * k2pp;

    k3gg -= k2gg;
    k3pp -= k2pp;
    k2gg *= 2.0;
    k2pp *= 2.0;
    longitudinalOdeAddOneFieldCached(rpart, fieldHarmD, btpar, ztemp1, xks, xku, efield,
                                     tgamma, ttheta, algebraOpt, &k2gg, &k2pp);

    tgamma += stpz * (k3gg + k2gg / 6.0);
    ttheta += stpz * (k3pp + k2pp / 6.0);

    gamma[ip] = tgamma;
    theta[ip] = ttheta;
}

__global__ void beamLongitudinalKernel(double *gamma,
                                       double *theta,
                                       const double *x,
                                       const double *y,
                                       const double *px,
                                       const double *py,
                                       const double *ez,
                                       const int *particleSlice,
                                       std::size_t npart,
                                       const GComplex *fields,
                                       GComplex * const *fieldData,
                                       const std::size_t *fieldOffset,
                                       const int *fieldNgrid,
                                       const int *fieldNslices,
                                       const int *fieldHarm,
                                       const double *fieldGridmax,
                                       const double *fieldDgrid,
                                       const double *fieldRtmp,
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
                                       double undGrady) {
    const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ip >= npart) {
        return;
    }

    double tgamma = gamma[ip];
    double ttheta = theta[ip] + autophase;
    const double awloc = undulatorAwLocal(x[ip], y[ip], undAx, undAy, undKx, undKy, undGradx, undGrady);
    const double btpar = 1.0 + px[ip] * px[ip] + py[ip] * py[ip] + aw * aw * awloc * awloc;
    const double efield = ez[ip];
    const int slice = particleSlice[ip];

    double k2gg = 0.0;
    double k2pp = 0.0;
    longitudinalOdeAdd(x, y, fields, fieldData, fieldOffset, fieldNgrid, fieldNslices, fieldHarm,
                       fieldGridmax, fieldDgrid, fieldRtmp, nfield, ip, slice, awloc,
                       btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    double stpz = 0.5 * delz;
    tgamma += stpz * k2gg;
    ttheta += stpz * k2pp;

    double k3gg = k2gg;
    double k3pp = k2pp;
    k2gg = 0.0;
    k2pp = 0.0;
    longitudinalOdeAdd(x, y, fields, fieldData, fieldOffset, fieldNgrid, fieldNslices, fieldHarm,
                       fieldGridmax, fieldDgrid, fieldRtmp, nfield, ip, slice, awloc,
                       btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    tgamma += stpz * (k2gg - k3gg);
    ttheta += stpz * (k2pp - k3pp);

    k3gg /= 6.0;
    k3pp /= 6.0;
    k2gg *= -0.5;
    k2pp *= -0.5;
    longitudinalOdeAdd(x, y, fields, fieldData, fieldOffset, fieldNgrid, fieldNslices, fieldHarm,
                       fieldGridmax, fieldDgrid, fieldRtmp, nfield, ip, slice, awloc,
                       btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    stpz = delz;
    tgamma += stpz * k2gg;
    ttheta += stpz * k2pp;

    k3gg -= k2gg;
    k3pp -= k2pp;
    k2gg *= 2.0;
    k2pp *= 2.0;
    longitudinalOdeAdd(x, y, fields, fieldData, fieldOffset, fieldNgrid, fieldNslices, fieldHarm,
                       fieldGridmax, fieldDgrid, fieldRtmp, nfield, ip, slice, awloc,
                       btpar, xks, xku, efield, tgamma, ttheta, &k2gg, &k2pp);

    tgamma += stpz * (k3gg + k2gg / 6.0);
    ttheta += stpz * (k3pp + k2pp / 6.0);

    gamma[ip] = tgamma;
    theta[ip] = ttheta;
}

} // namespace

struct BeamState {
    std::size_t nslice {0};
    std::size_t npart {0};
    std::size_t particleCapacity {0};
    std::size_t sliceCapacity {0};
    std::size_t fieldGridCapacity {0};
    std::size_t fieldMetaCapacity {0};
    std::size_t fieldPointerCapacity {0};
    std::size_t matrixCapacity {0};

    double *gamma {nullptr};
    double *theta {nullptr};
    double *x {nullptr};
    double *y {nullptr};
    double *px {nullptr};
    double *py {nullptr};
    double *ez {nullptr};

    int *sliceStart {nullptr};
    int *sliceCount {nullptr};
    int *particleSlice {nullptr};

    GComplex *radField {nullptr};
    GComplex **fieldData {nullptr};
    std::size_t *fieldOffset {nullptr};
    int *fieldNgrid {nullptr};
    int *fieldNslices {nullptr};
    int *fieldHarm {nullptr};
    double *fieldGridmax {nullptr};
    double *fieldDgrid {nullptr};
    double *fieldRtmp {nullptr};

    std::vector<GComplex *> hostFieldData;
    std::vector<std::size_t> hostFieldOffset;
    std::vector<int> hostFieldNgrid;
    std::vector<int> hostFieldNslices;
    std::vector<int> hostFieldHarm;
    std::vector<double> hostFieldGridmax;
    std::vector<double> hostFieldDgrid;
    std::vector<double> hostFieldRtmp;

    double *matrix4x4 {nullptr};
    BeamSliceDiagnostic *diagBeam {nullptr};
    GComplex *diagBunching {nullptr};
    std::size_t diagBeamCapacity {0};
    std::size_t diagBunchingCapacity {0};
};

namespace {

inline bool ensureBeamParticleBuffers(BeamState *state, std::size_t npart) {
    const bool grow = state->particleCapacity < npart;
    const bool missing = (state->gamma == nullptr) || (state->theta == nullptr) ||
                         (state->x == nullptr) || (state->y == nullptr) ||
                         (state->px == nullptr) || (state->py == nullptr) ||
                         (state->ez == nullptr) || (state->particleSlice == nullptr);

    if (!ensureDoubleBuffer(&state->gamma, &state->particleCapacity, npart, "cudaMalloc(beam gamma)")) return false;

    if (grow || missing) {
        const std::size_t allocCount = state->particleCapacity;
        trackedCudaFree(state->theta); state->theta = nullptr;
        trackedCudaFree(state->x); state->x = nullptr;
        trackedCudaFree(state->y); state->y = nullptr;
        trackedCudaFree(state->px); state->px = nullptr;
        trackedCudaFree(state->py); state->py = nullptr;
        trackedCudaFree(state->ez); state->ez = nullptr;
        trackedCudaFree(state->particleSlice); state->particleSlice = nullptr;
        std::size_t dummy = 0;
        if (!ensureDoubleBuffer(&state->theta, &dummy, allocCount, "cudaMalloc(beam theta)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->x, &dummy, allocCount, "cudaMalloc(beam x)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->y, &dummy, allocCount, "cudaMalloc(beam y)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->px, &dummy, allocCount, "cudaMalloc(beam px)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->py, &dummy, allocCount, "cudaMalloc(beam py)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->ez, &dummy, allocCount, "cudaMalloc(beam ez)")) return false;
        dummy = 0; if (!ensureIntBuffer(&state->particleSlice, &dummy, allocCount, "cudaMalloc(beam particleSlice)")) return false;
    }
    return true;
}

inline bool ensureBeamFieldBuffers(BeamState *state, std::size_t fieldGridPoints, std::size_t nfield) {
    const bool metaGrow = state->fieldMetaCapacity < nfield;
    const bool metaMissing = (state->fieldOffset == nullptr) || (state->fieldNgrid == nullptr) ||
                             (state->fieldNslices == nullptr) || (state->fieldHarm == nullptr) ||
                             (state->fieldGridmax == nullptr) || (state->fieldDgrid == nullptr) ||
                             (state->fieldRtmp == nullptr);

    if (!ensureComplexBuffer(&state->radField, &state->fieldGridCapacity, fieldGridPoints, "cudaMalloc(beam radiation fields)")) return false;
    if (!ensureSizeBuffer(&state->fieldOffset, &state->fieldMetaCapacity, nfield, "cudaMalloc(beam field offsets)")) return false;
    if (!ensureComplexPointerBuffer(&state->fieldData, &state->fieldPointerCapacity, nfield, "cudaMalloc(beam field pointer table)")) return false;

    state->hostFieldData.resize(nfield, nullptr);
    state->hostFieldOffset.resize(nfield, 0);
    state->hostFieldNgrid.resize(nfield, 0);
    state->hostFieldNslices.resize(nfield, 0);
    state->hostFieldHarm.resize(nfield, 0);
    state->hostFieldGridmax.resize(nfield, 0.0);
    state->hostFieldDgrid.resize(nfield, 0.0);
    state->hostFieldRtmp.resize(nfield, 0.0);

    if (metaGrow || metaMissing) {
        const std::size_t allocCount = state->fieldMetaCapacity;
        trackedCudaFree(state->fieldNgrid); state->fieldNgrid = nullptr;
        trackedCudaFree(state->fieldNslices); state->fieldNslices = nullptr;
        trackedCudaFree(state->fieldHarm); state->fieldHarm = nullptr;
        trackedCudaFree(state->fieldGridmax); state->fieldGridmax = nullptr;
        trackedCudaFree(state->fieldDgrid); state->fieldDgrid = nullptr;
        trackedCudaFree(state->fieldRtmp); state->fieldRtmp = nullptr;
        std::size_t dummy = 0;
        if (!ensureIntBuffer(&state->fieldNgrid, &dummy, allocCount, "cudaMalloc(beam field ngrid)")) return false;
        dummy = 0; if (!ensureIntBuffer(&state->fieldNslices, &dummy, allocCount, "cudaMalloc(beam field nslices)")) return false;
        dummy = 0; if (!ensureIntBuffer(&state->fieldHarm, &dummy, allocCount, "cudaMalloc(beam field harmonic)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->fieldGridmax, &dummy, allocCount, "cudaMalloc(beam field gridmax)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->fieldDgrid, &dummy, allocCount, "cudaMalloc(beam field dgrid)")) return false;
        dummy = 0; if (!ensureDoubleBuffer(&state->fieldRtmp, &dummy, allocCount, "cudaMalloc(beam field rtmp)")) return false;
    }
    return true;
}

} // namespace

BeamState *beamCreate() {
    return new BeamState();
}

void beamDestroy(BeamState *state) {
    if (state == nullptr) {
        return;
    }
    trackedCudaFree(state->gamma);
    trackedCudaFree(state->theta);
    trackedCudaFree(state->x);
    trackedCudaFree(state->y);
    trackedCudaFree(state->px);
    trackedCudaFree(state->py);
    trackedCudaFree(state->ez);
    trackedCudaFree(state->sliceStart);
    trackedCudaFree(state->sliceCount);
    trackedCudaFree(state->particleSlice);
    trackedCudaFree(state->radField);
    trackedCudaFree(state->fieldData);
    trackedCudaFree(state->fieldOffset);
    trackedCudaFree(state->fieldNgrid);
    trackedCudaFree(state->fieldNslices);
    trackedCudaFree(state->fieldHarm);
    trackedCudaFree(state->fieldGridmax);
    trackedCudaFree(state->fieldDgrid);
    trackedCudaFree(state->fieldRtmp);
    trackedCudaFree(state->matrix4x4);
    trackedCudaFree(state->diagBeam);
    trackedCudaFree(state->diagBunching);
    delete state;
}

bool beamEnsure(BeamState *state, std::size_t nslice, std::size_t npart) {
    if (state == nullptr) {
        g_lastError = "null CUDA beam state";
        return false;
    }
    if (npart > static_cast<std::size_t>(INT_MAX)) {
        g_lastError = "CUDA beam particle count exceeds supported int range";
        return false;
    }
    if (!ensureBeamParticleBuffers(state, npart)) return false;
    const bool sliceGrow = state->sliceCapacity < nslice;
    const bool sliceMissing = (state->sliceStart == nullptr) || (state->sliceCount == nullptr);
    if (!ensureIntBuffer(&state->sliceStart, &state->sliceCapacity, nslice, "cudaMalloc(beam sliceStart)")) return false;
    if (sliceGrow || sliceMissing) {
        trackedCudaFree(state->sliceCount);
        state->sliceCount = nullptr;
        std::size_t dummy = 0;
        if (!ensureIntBuffer(&state->sliceCount, &dummy, state->sliceCapacity, "cudaMalloc(beam sliceCount)")) return false;
    }
    state->nslice = nslice;
    state->npart = npart;
    return true;
}

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
                         std::size_t npart) {
    if ((state == nullptr) || (state->particleCapacity < npart) || (state->sliceCapacity < nslice)) {
        g_lastError = "CUDA beam upload called before buffers were initialized";
        return false;
    }
    if (npart > 0) {
        if (!setError("cudaMemcpy(beam gamma)", cudaMemcpy(state->gamma, gamma, npart * sizeof(double), cudaMemcpyHostToDevice))) return false;
        if (!setError("cudaMemcpy(beam theta)", cudaMemcpy(state->theta, theta, npart * sizeof(double), cudaMemcpyHostToDevice))) return false;
        if (!setError("cudaMemcpy(beam x)", cudaMemcpy(state->x, x, npart * sizeof(double), cudaMemcpyHostToDevice))) return false;
        if (!setError("cudaMemcpy(beam y)", cudaMemcpy(state->y, y, npart * sizeof(double), cudaMemcpyHostToDevice))) return false;
        if (!setError("cudaMemcpy(beam px)", cudaMemcpy(state->px, px, npart * sizeof(double), cudaMemcpyHostToDevice))) return false;
        if (!setError("cudaMemcpy(beam py)", cudaMemcpy(state->py, py, npart * sizeof(double), cudaMemcpyHostToDevice))) return false;
        if (!setError("cudaMemcpy(beam particleSlice)", cudaMemcpy(state->particleSlice, particleSlice, npart * sizeof(int), cudaMemcpyHostToDevice))) return false;
    }
    if (nslice > 0) {
        if (!setError("cudaMemcpy(beam sliceStart)", cudaMemcpy(state->sliceStart, sliceStart, nslice * sizeof(int), cudaMemcpyHostToDevice))) return false;
        if (!setError("cudaMemcpy(beam sliceCount)", cudaMemcpy(state->sliceCount, sliceCount, nslice * sizeof(int), cudaMemcpyHostToDevice))) return false;
    }
    state->nslice = nslice;
    state->npart = npart;
    return true;
}

bool beamDownloadParticles(BeamState *state,
                           double *gamma,
                           double *theta,
                           double *x,
                           double *y,
                           double *px,
                           double *py,
                           std::size_t npart) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam download called before buffers were initialized";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    if (!setError("cudaMemcpy(beam gamma D2H)", cudaMemcpy(gamma, state->gamma, npart * sizeof(double), cudaMemcpyDeviceToHost))) return false;
    if (!setError("cudaMemcpy(beam theta D2H)", cudaMemcpy(theta, state->theta, npart * sizeof(double), cudaMemcpyDeviceToHost))) return false;
    if (!setError("cudaMemcpy(beam x D2H)", cudaMemcpy(x, state->x, npart * sizeof(double), cudaMemcpyDeviceToHost))) return false;
    if (!setError("cudaMemcpy(beam y D2H)", cudaMemcpy(y, state->y, npart * sizeof(double), cudaMemcpyDeviceToHost))) return false;
    if (!setError("cudaMemcpy(beam px D2H)", cudaMemcpy(px, state->px, npart * sizeof(double), cudaMemcpyDeviceToHost))) return false;
    if (!setError("cudaMemcpy(beam py D2H)", cudaMemcpy(py, state->py, npart * sizeof(double), cudaMemcpyDeviceToHost))) return false;
    return true;
}

bool beamUploadEz(BeamState *state, const double *ez, std::size_t npart) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam ez upload called before buffers were initialized";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    return setError("cudaMemcpy(beam ez)", cudaMemcpy(state->ez, ez, npart * sizeof(double), cudaMemcpyHostToDevice));
}

bool beamClearEz(BeamState *state, std::size_t npart) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam ez clear called before buffers were initialized";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    if (!setError("cudaMemset(beam ez)", cudaMemset(state->ez, 0, npart * sizeof(double)))) return false;
    return true;
}

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
                      std::size_t nfield) {
    if (!beamEnsureFields(state, fieldGridPoints, nfield)) return false;
    if (!beamUploadFieldDataAt(state, 0, fields, fieldGridPoints)) return false;
    if (!beamUploadFieldMetadata(state,
                                 fieldOffset,
                                 fieldNgrid,
                                 fieldNslices,
                                 fieldHarm,
                                 fieldGridmax,
                                 fieldDgrid,
                                 fieldRtmp,
                                 nfield)) return false;
    for (std::size_t ifld = 0; ifld < nfield; ++ifld) {
        if (!beamUseInternalFieldData(state, ifld, fieldOffset[ifld])) return false;
    }
    return true;
}

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
                            double undGrady) {
    if ((state == nullptr) || (beamState == nullptr)) {
        g_lastError = "null CUDA state in SoA source deposition";
        return false;
    }
    if (!validateGridAndBatch(state, ngrid, nslice, "CUDA SoA source deposition")) {
        return false;
    }
    if ((beamState->sliceCapacity < nslice) || (beamState->sliceStart == nullptr) ||
        (beamState->sliceCount == nullptr) || (beamState->gamma == nullptr) ||
        (beamState->theta == nullptr) || (beamState->x == nullptr) || (beamState->y == nullptr)) {
        g_lastError = "CUDA SoA source deposition called before beam buffers were initialized";
        return false;
    }
    if (nslice == 0) {
        return true;
    }
    if (!ensureDoubleBuffer(&state->fftSliceScale,
                            &state->fftSliceScaleCapacity,
                            nslice,
                            "cudaMalloc(fft slice scales)")) {
        return false;
    }
    if (!setError("cudaMemcpy(fft slice scales)",
                  cudaMemcpy(state->fftSliceScale,
                             sliceScale,
                             nslice * sizeof(double),
                             cudaMemcpyHostToDevice))) {
        return false;
    }

    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * ngrid;
    const int block = sourceDepositionBlockSize();
    const int grid = static_cast<int>(nslice);
    buildSourceFromSoAKernel<<<grid, block>>>(beamState->gamma,
                                              beamState->theta,
                                              beamState->x,
                                              beamState->y,
                                              beamState->sliceStart,
                                              beamState->sliceCount,
                                              state->fftSliceScale,
                                              state->source,
                                              ngrid,
                                              ngrid2,
                                              nslice,
                                              gridmax,
                                              dgrid,
                                              harm,
                                              undAx,
                                              undAy,
                                              undKx,
                                              undKy,
                                              undGradx,
                                              undGrady);
    return checkKernel("SoA source deposition kernel");
}

bool beamEnsureFields(BeamState *state,
                      std::size_t fieldGridPoints,
                      std::size_t nfield) {
    if (state == nullptr) {
        g_lastError = "null CUDA beam state";
        return false;
    }
    return ensureBeamFieldBuffers(state, fieldGridPoints, nfield);
}

bool beamUploadFieldMetadata(BeamState *state,
                             const std::size_t *fieldOffset,
                             const int *fieldNgrid,
                             const int *fieldNslices,
                             const int *fieldHarm,
                             const double *fieldGridmax,
                             const double *fieldDgrid,
                             const double *fieldRtmp,
                             std::size_t nfield) {
    if (state == nullptr) {
        g_lastError = "null CUDA beam state";
        return false;
    }
    if (state->fieldMetaCapacity < nfield) {
        g_lastError = "CUDA beam field metadata upload called before buffers were initialized";
        return false;
    }
    if (nfield == 0) {
        return true;
    }
    if ((fieldOffset == nullptr) || (fieldNgrid == nullptr) || (fieldNslices == nullptr) ||
        (fieldHarm == nullptr) || (fieldGridmax == nullptr) || (fieldDgrid == nullptr) ||
        (fieldRtmp == nullptr)) {
        g_lastError = "null CUDA beam field metadata host pointer";
        return false;
    }
    state->hostFieldOffset.assign(fieldOffset, fieldOffset + nfield);
    state->hostFieldNgrid.assign(fieldNgrid, fieldNgrid + nfield);
    state->hostFieldNslices.assign(fieldNslices, fieldNslices + nfield);
    state->hostFieldHarm.assign(fieldHarm, fieldHarm + nfield);
    state->hostFieldGridmax.assign(fieldGridmax, fieldGridmax + nfield);
    state->hostFieldDgrid.assign(fieldDgrid, fieldDgrid + nfield);
    state->hostFieldRtmp.assign(fieldRtmp, fieldRtmp + nfield);
    if (state->hostFieldData.size() < nfield) {
        state->hostFieldData.resize(nfield, nullptr);
    }
    if (!setError("cudaMemcpy(beam field offsets)", cudaMemcpy(state->fieldOffset, fieldOffset, nfield * sizeof(std::size_t), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(beam field ngrid)", cudaMemcpy(state->fieldNgrid, fieldNgrid, nfield * sizeof(int), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(beam field nslices)", cudaMemcpy(state->fieldNslices, fieldNslices, nfield * sizeof(int), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(beam field harmonic)", cudaMemcpy(state->fieldHarm, fieldHarm, nfield * sizeof(int), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(beam field gridmax)", cudaMemcpy(state->fieldGridmax, fieldGridmax, nfield * sizeof(double), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(beam field dgrid)", cudaMemcpy(state->fieldDgrid, fieldDgrid, nfield * sizeof(double), cudaMemcpyHostToDevice))) return false;
    if (!setError("cudaMemcpy(beam field rtmp)", cudaMemcpy(state->fieldRtmp, fieldRtmp, nfield * sizeof(double), cudaMemcpyHostToDevice))) return false;
    return true;
}

bool beamUploadFieldDataAt(BeamState *state,
                           std::size_t dstOffset,
                           const std::complex<double> *fields,
                           std::size_t fieldGridPoints) {
    if (state == nullptr) {
        g_lastError = "null CUDA beam state";
        return false;
    }
    if (fieldGridPoints == 0) {
        return true;
    }
    if ((fields == nullptr) || (state->radField == nullptr) || (dstOffset + fieldGridPoints > state->fieldGridCapacity)) {
        g_lastError = "CUDA beam field data upload called before buffers were initialized";
        return false;
    }
    return setError("cudaMemcpy(beam radiation fields)",
                    cudaMemcpy(state->radField + dstOffset,
                               reinterpret_cast<const GComplex *>(fields),
                               fieldGridPoints * sizeof(GComplex),
                               cudaMemcpyHostToDevice));
}

namespace {

bool setBeamFieldPointer(BeamState *beamState, std::size_t fieldIndex, GComplex *ptr, const char *context) {
    if (beamState == nullptr) {
        g_lastError = std::string(context) + ": null CUDA beam state";
        return false;
    }
    if ((beamState->fieldData == nullptr) || (fieldIndex >= beamState->fieldPointerCapacity)) {
        g_lastError = std::string(context) + ": field pointer table is not initialized";
        return false;
    }
    if (beamState->hostFieldData.size() <= fieldIndex) {
        beamState->hostFieldData.resize(fieldIndex + 1, nullptr);
    }
    beamState->hostFieldData[fieldIndex] = ptr;
    return setError(context,
                    cudaMemcpy(beamState->fieldData + fieldIndex,
                               &ptr,
                               sizeof(GComplex *),
                               cudaMemcpyHostToDevice));
}

} // namespace

bool beamUseInternalFieldData(BeamState *beamState,
                              std::size_t fieldIndex,
                              std::size_t dstOffset) {
    if (beamState == nullptr) {
        g_lastError = "null CUDA beam state in internal field binding";
        return false;
    }
    if ((beamState->radField == nullptr) || (dstOffset >= beamState->fieldGridCapacity)) {
        g_lastError = "internal beam field binding requested an out-of-range destination";
        return false;
    }
    return setBeamFieldPointer(beamState,
                               fieldIndex,
                               beamState->radField + dstOffset,
                               "cudaMemcpy(beam internal field pointer)");
}

bool beamBindFFTField(BeamState *beamState,
                      std::size_t fieldIndex,
                      State *fftState,
                      unsigned int ngrid,
                      std::size_t batchSize) {
    if ((beamState == nullptr) || (fftState == nullptr)) {
        g_lastError = "null CUDA state in FFT field binding";
        return false;
    }
    if (!validateGridAndBatch(fftState, ngrid, batchSize, "CUDA FFT field binding")) {
        return false;
    }
    return setBeamFieldPointer(beamState,
                               fieldIndex,
                               fftState->field,
                               "cudaMemcpy(beam external FFT field pointer)");
}

bool copyFFTFieldsToBeam(BeamState *beamState,
                         std::size_t dstOffset,
                         State *fftState,
                         unsigned int ngrid,
                         std::size_t batchSize) {
    if ((beamState == nullptr) || (fftState == nullptr)) {
        g_lastError = "null CUDA state in FFT-to-beam field copy";
        return false;
    }
    if (!validateGridAndBatch(fftState, ngrid, batchSize, "CUDA FFT-to-beam field copy")) {
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t total = n * batchSize;
    if ((beamState->radField == nullptr) || (dstOffset + total > beamState->fieldGridCapacity)) {
        g_lastError = "CUDA FFT-to-beam field copy destination is too small";
        return false;
    }
    return setError("cudaMemcpy(FFT field D2D -> beam fields)",
                    cudaMemcpy(beamState->radField + dstOffset,
                               fftState->field,
                               total * sizeof(GComplex),
                               cudaMemcpyDeviceToDevice));
}


namespace {

bool ensureBeamDiagnostics(BeamState *state, std::size_t nslice, std::size_t nbunch) {
    if (state == nullptr) {
        g_lastError = "null CUDA beam state in diagnostics";
        return false;
    }
    if (state->diagBeamCapacity < nslice) {
        trackedCudaFree(state->diagBeam);
        state->diagBeam = nullptr;
        state->diagBeamCapacity = 0;
        if (nslice > 0) {
            if (!setError("cudaMalloc(beam diagnostics)",
                          trackedCudaMalloc(reinterpret_cast<void **>(&state->diagBeam),
                                     nslice * sizeof(BeamSliceDiagnostic),
                                     "cudaMalloc(beam diagnostics)"))) {
                return false;
            }
            state->diagBeamCapacity = nslice;
        }
    }
    if (state->diagBunchingCapacity < nbunch) {
        trackedCudaFree(state->diagBunching);
        state->diagBunching = nullptr;
        state->diagBunchingCapacity = 0;
        if (nbunch > 0) {
            if (!setError("cudaMalloc(beam bunching diagnostics)",
                          trackedCudaMalloc(reinterpret_cast<void **>(&state->diagBunching),
                                     nbunch * sizeof(GComplex),
                                     "cudaMalloc(beam bunching diagnostics)"))) {
                return false;
            }
            state->diagBunchingCapacity = nbunch;
        }
    }
    return true;
}

bool ensureFieldDiagnostics(State *state, std::size_t nslice) {
    if (state == nullptr) {
        g_lastError = "null CUDA FFT state in diagnostics";
        return false;
    }
    if (state->diagFieldCapacity >= nslice) {
        return true;
    }
    trackedCudaFree(state->diagField);
    state->diagField = nullptr;
    state->diagFieldCapacity = 0;
    if (nslice == 0) {
        return true;
    }
    if (!setError("cudaMalloc(field diagnostics)",
                  trackedCudaMalloc(reinterpret_cast<void **>(&state->diagField),
                             nslice * sizeof(FieldSliceDiagnostic),
                             "cudaMalloc(field diagnostics)"))) {
        return false;
    }
    state->diagFieldCapacity = nslice;
    return true;
}

} // namespace

bool beamComputeSliceDiagnostics(BeamState *state,
                                 int nharm,
                                 BeamSliceDiagnostic *hostSliceDiagnostics,
                                 std::complex<double> *hostBunching) {
    if ((state == nullptr) || (hostSliceDiagnostics == nullptr)) {
        g_lastError = "null argument in CUDA beam diagnostics";
        return false;
    }
    if ((state->sliceStart == nullptr) || (state->sliceCount == nullptr) ||
        (state->gamma == nullptr) || (state->theta == nullptr) ||
        (state->x == nullptr) || (state->y == nullptr) ||
        (state->px == nullptr) || (state->py == nullptr)) {
        g_lastError = "CUDA beam diagnostics called before beam buffers were initialized";
        return false;
    }
    const std::size_t nslice = state->nslice;
    const std::size_t nbunch = (nharm > 0) ? nslice * static_cast<std::size_t>(nharm) : 0;
    if ((nbunch > 0) && (hostBunching == nullptr)) {
        g_lastError = "CUDA beam diagnostics requested bunching without a host output buffer";
        return false;
    }
    if (!ensureBeamDiagnostics(state, nslice, nbunch)) {
        return false;
    }
    if (nslice == 0) {
        return true;
    }

    const int block = 256;
    const std::size_t momentsSmem = static_cast<std::size_t>(22) * block * sizeof(double);
    beamMomentsDiagnosticKernel<<<static_cast<int>(nslice), block, momentsSmem>>>(state->gamma,
                                                                                  state->theta,
                                                                                  state->x,
                                                                                  state->y,
                                                                                  state->px,
                                                                                  state->py,
                                                                                  state->sliceStart,
                                                                                  state->sliceCount,
                                                                                  nslice,
                                                                                  state->diagBeam);
    if (!checkKernel("beam diagnostics moments kernel")) return false;

    if (nbunch > 0) {
        const dim3 grid(static_cast<unsigned int>(nslice), static_cast<unsigned int>(nharm), 1);
        const std::size_t bunchSmem = static_cast<std::size_t>(2) * block * sizeof(double);
        beamBunchingDiagnosticKernel<<<grid, block, bunchSmem>>>(state->theta,
                                                                 state->sliceStart,
                                                                 state->sliceCount,
                                                                 nslice,
                                                                 nharm,
                                                                 state->diagBunching);
        if (!checkKernel("beam diagnostics bunching kernel")) return false;
    }

    if (!setError("cudaMemcpy(beam diagnostics D2H)",
                  cudaMemcpy(hostSliceDiagnostics,
                             state->diagBeam,
                             nslice * sizeof(BeamSliceDiagnostic),
                             cudaMemcpyDeviceToHost))) {
        return false;
    }
    if (nbunch > 0) {
        if (!setError("cudaMemcpy(beam bunching diagnostics D2H)",
                      cudaMemcpy(reinterpret_cast<GComplex *>(hostBunching),
                                 state->diagBunching,
                                 nbunch * sizeof(GComplex),
                                 cudaMemcpyDeviceToHost))) {
            return false;
        }
    }
    return true;
}

bool fftFieldComputeSliceDiagnostics(State *state,
                                     unsigned int ngrid,
                                     std::size_t batchSize,
                                     bool includeFftMoments,
                                     FieldSliceDiagnostic *hostSliceDiagnostics) {
    if ((state == nullptr) || (hostSliceDiagnostics == nullptr)) {
        g_lastError = "null argument in CUDA field diagnostics";
        return false;
    }
    if (!validateGridAndBatch(state, ngrid, batchSize, "CUDA field diagnostics")) {
        return false;
    }
    if (!ensureFieldDiagnostics(state, batchSize)) {
        return false;
    }
    if (batchSize == 0) {
        return true;
    }

    const int block = 256;
    const std::size_t ngrid2 = static_cast<std::size_t>(ngrid) * ngrid;
    const std::size_t smem = static_cast<std::size_t>(7) * block * sizeof(double);
    fftFieldDiagnosticKernel<<<static_cast<int>(batchSize), block, smem>>>(state->field,
                                                                           ngrid,
                                                                           ngrid2,
                                                                           batchSize,
                                                                           state->diagField);
    if (!checkKernel("field diagnostics kernel")) return false;

    if (includeFftMoments) {
        const std::size_t total = ngrid2 * batchSize;
        if (!setError("cudaMemcpy(field diagnostics FFT staging D2D)",
                      cudaMemcpy(state->source,
                                 state->field,
                                 total * sizeof(GComplex),
                                 cudaMemcpyDeviceToDevice))) {
            return false;
        }
        if (!setCufftError("cufftExecZ2Z(field diagnostics forward FFT)",
                           cufftExecZ2Z(state->fftPlan,
                                        reinterpret_cast<cufftDoubleComplex *>(state->source),
                                        reinterpret_cast<cufftDoubleComplex *>(state->source),
                                        CUFFT_FORWARD))) {
            return false;
        }
        const std::size_t farSmem = static_cast<std::size_t>(5) * block * sizeof(double);
        fftFieldFarfieldDiagnosticKernel<<<static_cast<int>(batchSize), block, farSmem>>>(state->source,
                                                                                          ngrid,
                                                                                          ngrid2,
                                                                                          batchSize,
                                                                                          state->diagField);
        if (!checkKernel("field far-field diagnostics kernel")) return false;
    }

    return setError("cudaMemcpy(field diagnostics D2H)",
                    cudaMemcpy(hostSliceDiagnostics,
                               state->diagField,
                               batchSize * sizeof(FieldSliceDiagnostic),
                               cudaMemcpyDeviceToHost));
}

bool beamTrackTransverse(BeamState *state,
                         std::size_t npart,
                         double delz,
                         double aw,
                         double qx,
                         double qy,
                         double xoff,
                         double yoff,
                         int modeX,
                         int modeY) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam transverse track called before buffers were initialized";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    const int block = beamTransverseBlockSize();
    const int grid = static_cast<int>((npart + block - 1) / block);

    if (fastKernelsEnabled()) {
        if (symmetricTransverseEnabled() &&
            (modeX == TRACK_FOCUS) && (modeY == TRACK_FOCUS) &&
            (qx > 0.0) && (qy > 0.0)) {
            if (qx == qy) {
                beamTrackTransverseFocusSymmetricKernel<<<grid, block>>>(state->gamma,
                                                                         state->x,
                                                                         state->y,
                                                                         state->px,
                                                                         state->py,
                                                                         npart,
                                                                         delz,
                                                                         aw,
                                                                         qx,
                                                                         xoff,
                                                                         yoff);
                return checkKernel("symmetric focus beam transverse tracking kernel");
            }
        }
#define LAUNCH_TRANSVERSE(MX, MY) \
        beamTrackTransverseKernelT<MX, MY><<<grid, block>>>(state->gamma, \
                                                            state->x, \
                                                            state->y, \
                                                            state->px, \
                                                            state->py, \
                                                            npart, \
                                                            delz, \
                                                            aw, \
                                                            qx, \
                                                            qy, \
                                                            xoff, \
                                                            yoff)
        if (modeX == TRACK_DRIFT && modeY == TRACK_DRIFT) { LAUNCH_TRANSVERSE(TRACK_DRIFT, TRACK_DRIFT); }
        else if (modeX == TRACK_DRIFT && modeY == TRACK_FOCUS) { LAUNCH_TRANSVERSE(TRACK_DRIFT, TRACK_FOCUS); }
        else if (modeX == TRACK_DRIFT) { LAUNCH_TRANSVERSE(TRACK_DRIFT, TRACK_DEFOCUS); }
        else if (modeX == TRACK_FOCUS && modeY == TRACK_DRIFT) { LAUNCH_TRANSVERSE(TRACK_FOCUS, TRACK_DRIFT); }
        else if (modeX == TRACK_FOCUS && modeY == TRACK_FOCUS) { LAUNCH_TRANSVERSE(TRACK_FOCUS, TRACK_FOCUS); }
        else if (modeX == TRACK_FOCUS) { LAUNCH_TRANSVERSE(TRACK_FOCUS, TRACK_DEFOCUS); }
        else if (modeY == TRACK_DRIFT) { LAUNCH_TRANSVERSE(TRACK_DEFOCUS, TRACK_DRIFT); }
        else if (modeY == TRACK_FOCUS) { LAUNCH_TRANSVERSE(TRACK_DEFOCUS, TRACK_FOCUS); }
        else { LAUNCH_TRANSVERSE(TRACK_DEFOCUS, TRACK_DEFOCUS); }
#undef LAUNCH_TRANSVERSE
        return checkKernel("specialized beam transverse tracking kernel");
    }

    beamTrackTransverseKernel<<<grid, block>>>(state->gamma,
                                               state->x,
                                               state->y,
                                               state->px,
                                               state->py,
                                               npart,
                                               delz,
                                               aw,
                                               qx,
                                               qy,
                                               xoff,
                                               yoff,
                                               modeX,
                                               modeY);
    return checkKernel("beam transverse tracking kernel");
}

bool beamApplyCorrector(BeamState *state, std::size_t npart, double cx, double cy) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam corrector called before buffers were initialized";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    const int block = 256;
    const int grid = static_cast<int>((npart + block - 1) / block);
    beamCorrectorKernel<<<grid, block>>>(state->px, state->py, npart, cx, cy);
    return checkKernel("beam corrector kernel");
}

bool beamApplyChicaneMatrix(BeamState *state,
                            std::size_t npart,
                            const double *matrix4x4) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam chicane called before buffers were initialized";
        return false;
    }
    if (!ensureDoubleBuffer(&state->matrix4x4, &state->matrixCapacity, 16, "cudaMalloc(beam chicane matrix)")) return false;
    if (!setError("cudaMemcpy(beam chicane matrix)", cudaMemcpy(state->matrix4x4, matrix4x4, 16 * sizeof(double), cudaMemcpyHostToDevice))) return false;
    if (npart == 0) {
        return true;
    }
    const int block = 256;
    const int grid = static_cast<int>((npart + block - 1) / block);
    beamChicaneMatrixKernel<<<grid, block>>>(state->gamma,
                                             state->x,
                                             state->y,
                                             state->px,
                                             state->py,
                                             npart,
                                             state->matrix4x4);
    return checkKernel("beam chicane matrix kernel");
}

bool beamApplyR56(BeamState *state, std::size_t npart, double r56, double gamma0) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam R56 called before buffers were initialized";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    const int block = 256;
    const int grid = static_cast<int>((npart + block - 1) / block);
    beamR56Kernel<<<grid, block>>>(state->gamma, state->theta, npart, r56, gamma0);
    return checkKernel("beam R56 kernel");
}

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
                             double undGrady) {
    if ((state == nullptr) || (state->particleCapacity < npart)) {
        g_lastError = "CUDA beam longitudinal advance called before buffers were initialized";
        return false;
    }
    if ((nfield > 0) && ((state->fieldData == nullptr) ||
                          (state->fieldMetaCapacity < nfield) ||
                          (state->fieldPointerCapacity < nfield))) {
        g_lastError = "CUDA beam longitudinal advance called before fields were bound";
        return false;
    }
    if (npart == 0) {
        return true;
    }
    if (xks == 0.0) {
        g_lastError = "CUDA beam longitudinal advance received xks=0";
        return false;
    }
    const int block = beamLongitudinalBlockSize();
    const int grid = static_cast<int>((npart + block - 1) / block);

    if (fastKernelsEnabled() && (nfield == 1) &&
        (state->hostFieldData.size() >= 1) &&
        (state->hostFieldNgrid.size() >= 1) &&
        (state->hostFieldNslices.size() >= 1) &&
        (state->hostFieldHarm.size() >= 1) &&
        (state->hostFieldGridmax.size() >= 1) &&
        (state->hostFieldDgrid.size() >= 1) &&
        (state->hostFieldRtmp.size() >= 1) &&
        (state->hostFieldData[0] != nullptr)) {
        if (cacheLongitudinalInterpolationEnabled()) {
            beamLongitudinalOneFieldCachedInterpKernel<<<grid, block>>>(state->gamma,
                                                                        state->theta,
                                                                        state->x,
                                                                        state->y,
                                                                        state->px,
                                                                        state->py,
                                                                        state->ez,
                                                                        state->particleSlice,
                                                                        npart,
                                                                        state->hostFieldData[0],
                                                                        state->hostFieldNgrid[0],
                                                                        state->hostFieldNslices[0],
                                                                        state->hostFieldHarm[0],
                                                                        state->hostFieldGridmax[0],
                                                                        state->hostFieldDgrid[0],
                                                                        state->hostFieldRtmp[0],
                                                                        delz,
                                                                        xks,
                                                                        xku,
                                                                        aw,
                                                                        autophase,
                                                                        undAx,
                                                                        undAy,
                                                                        undKx,
                                                                        undKy,
                                                                        undGradx,
                                                                        undGrady,
                                                                        longitudinalAlgebraOptimizedEnabled());
            return checkKernel("cached-interpolation one-field beam longitudinal RK4 kernel");
        }

        beamLongitudinalOneFieldKernel<<<grid, block>>>(state->gamma,
                                                        state->theta,
                                                        state->x,
                                                        state->y,
                                                        state->px,
                                                        state->py,
                                                        state->ez,
                                                        state->particleSlice,
                                                        npart,
                                                        state->hostFieldData[0],
                                                        state->hostFieldNgrid[0],
                                                        state->hostFieldNslices[0],
                                                        state->hostFieldHarm[0],
                                                        state->hostFieldGridmax[0],
                                                        state->hostFieldDgrid[0],
                                                        state->hostFieldRtmp[0],
                                                        delz,
                                                        xks,
                                                        xku,
                                                        aw,
                                                        autophase,
                                                        undAx,
                                                        undAy,
                                                        undKx,
                                                        undKy,
                                                        undGradx,
                                                        undGrady);
        return checkKernel("specialized one-field beam longitudinal RK4 kernel");
    }

    beamLongitudinalKernel<<<grid, block>>>(state->gamma,
                                            state->theta,
                                            state->x,
                                            state->y,
                                            state->px,
                                            state->py,
                                            state->ez,
                                            state->particleSlice,
                                            npart,
                                            state->radField,
                                            state->fieldData,
                                            state->fieldOffset,
                                            state->fieldNgrid,
                                            state->fieldNslices,
                                            state->fieldHarm,
                                            state->fieldGridmax,
                                            state->fieldDgrid,
                                            state->fieldRtmp,
                                            nfield,
                                            delz,
                                            xks,
                                            xku,
                                            aw,
                                            autophase,
                                            undAx,
                                            undAy,
                                            undKx,
                                            undKy,
                                            undGradx,
                                            undGrady);
    return checkKernel("beam longitudinal RK4 kernel");
}


void setMemoryAuditContext(int worldRank, int mpiSize) {
    MemoryAuditState &audit = memoryAudit();
    audit.worldRank = worldRank;
    audit.mpiSize = mpiSize;
    int device = -1;
    cudaGetDevice(&device);
    audit.device = device;
}

bool memoryAuditEnabled() {
    return memoryAudit().enabled;
}

void printMemoryAuditSummary() {
    MemoryAuditState &audit = memoryAudit();
    if (!audit.enabled) {
        return;
    }
    int device = -1;
    cudaGetDevice(&device);
    audit.device = device;
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "GENESIS CUDA memory audit: rank=" << audit.worldRank
       << "/" << audit.mpiSize
       << " device=" << audit.device
       << " currentMiB=" << bytesToMiB(audit.currentBytes)
       << " peakMiB=" << bytesToMiB(audit.peakBytes)
       << " totalAllocMiB=" << bytesToMiB(audit.totalAllocatedBytes)
       << " totalFreeMiB=" << bytesToMiB(audit.totalFreedBytes)
       << " allocs=" << audit.allocations
       << " frees=" << audit.frees
       << " unknownFrees=" << audit.unknownFrees
       << " active=" << audit.active.size()
       << " cufftPlanCreates=" << audit.cufftPlanCreates
       << " cufftPlanDestroys=" << audit.cufftPlanDestroys
       << " cufftWorkspaceEstimatePeakMiB=" << bytesToMiB(audit.cufftWorkspaceEstimatePeak)
       << " cufftWorkspaceEstimateCurrentMiB=" << bytesToMiB(audit.cufftWorkspaceEstimateCurrent);
    std::cout << os.str() << std::endl;

    const char *topEnv = std::getenv("GENESIS_CUDA_MEMORY_AUDIT_TOP");
    int top = 8;
    if (topEnv != nullptr && topEnv[0] != '\0') {
        top = std::max(0, std::atoi(topEnv));
    }
    if (top == 0) {
        return;
    }
    std::vector<std::pair<std::string, MemoryAuditCategory>> cats(audit.categories.begin(), audit.categories.end());
    std::sort(cats.begin(), cats.end(), [](const auto &a, const auto &b) {
        if (a.second.peakBytes != b.second.peakBytes) {
            return a.second.peakBytes > b.second.peakBytes;
        }
        return a.first < b.first;
    });
    for (int i = 0; i < top && i < static_cast<int>(cats.size()); ++i) {
        const auto &entry = cats[static_cast<std::size_t>(i)];
        const MemoryAuditCategory &cat = entry.second;
        std::ostringstream line;
        line << std::fixed << std::setprecision(3);
        line << "GENESIS CUDA memory audit category: rank=" << audit.worldRank
             << " device=" << audit.device
             << " name=\"" << entry.first << "\""
             << " currentMiB=" << bytesToMiB(cat.currentBytes)
             << " peakMiB=" << bytesToMiB(cat.peakBytes)
             << " totalAllocMiB=" << bytesToMiB(cat.totalAllocatedBytes)
             << " allocs=" << cat.allocations
             << " frees=" << cat.frees;
        std::cout << line.str() << std::endl;
    }
}

} // namespace genesis_cuda
