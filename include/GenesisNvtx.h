#ifndef GENESIS_NVTX_H
#define GENESIS_NVTX_H

// Lightweight NVTX instrumentation for Genesis CUDA profiling.
//
// The implementation dynamically loads libnvToolsExt at runtime, so the
// project does not need a hard link dependency on the NVTX library.  If NVTX is
// unavailable, all ranges become no-ops.  Ranges can be disabled at runtime by
// setting GENESIS_CUDA_NVTX=0.

class GenesisNvtxRange {
public:
    explicit GenesisNvtxRange(const char *name);
    ~GenesisNvtxRange();

    GenesisNvtxRange(const GenesisNvtxRange &) = delete;
    GenesisNvtxRange &operator=(const GenesisNvtxRange &) = delete;

private:
    bool active_ {false};
};

namespace genesis_nvtx {
bool enabled();
void rangePush(const char *name);
void rangePop();
}

#if defined(GENESIS_NVTX)
#define GENESIS_NVTX_CONCAT_INNER(a,b) a##b
#define GENESIS_NVTX_CONCAT(a,b) GENESIS_NVTX_CONCAT_INNER(a,b)
#define GENESIS_NVTX_RANGE(name) GenesisNvtxRange GENESIS_NVTX_CONCAT(genesis_nvtx_range_, __LINE__)(name)
#else
#define GENESIS_NVTX_RANGE(name) do { (void)sizeof(name); } while (0)
#endif

#endif // GENESIS_NVTX_H
