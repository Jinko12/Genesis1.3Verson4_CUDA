#include "GenesisNvtx.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(GENESIS_NVTX)
#if defined(_WIN32)
#define GENESIS_NVTX_NO_DLOPEN 1
#else
#include <dlfcn.h>
#endif
#endif

namespace {

bool envFlagEnabled(const char *name, bool defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    if ((std::strcmp(value, "0") == 0) ||
        (std::strcmp(value, "false") == 0) ||
        (std::strcmp(value, "FALSE") == 0) ||
        (std::strcmp(value, "off") == 0) ||
        (std::strcmp(value, "OFF") == 0) ||
        (std::strcmp(value, "no") == 0) ||
        (std::strcmp(value, "NO") == 0)) {
        return false;
    }
    return true;
}

#if defined(GENESIS_NVTX) && !defined(GENESIS_NVTX_NO_DLOPEN)
struct NvtxApi {
    using RangePushA = int (*)(const char *);
    using RangePop = int (*)();

    void *handle {nullptr};
    RangePushA push {nullptr};
    RangePop pop {nullptr};
    bool attempted {false};
    bool available {false};
};

NvtxApi &api()
{
    static NvtxApi instance;
    static std::once_flag flag;
    std::call_once(flag, []() {
        instance.attempted = true;
        const char *candidates[] = {
            "libnvToolsExt.so.1",
            "libnvToolsExt.so",
            nullptr
        };
        for (int i = 0; candidates[i] != nullptr; ++i) {
            instance.handle = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
            if (instance.handle != nullptr) {
                break;
            }
        }
        if (instance.handle == nullptr) {
            return;
        }
        instance.push = reinterpret_cast<NvtxApi::RangePushA>(dlsym(instance.handle, "nvtxRangePushA"));
        instance.pop = reinterpret_cast<NvtxApi::RangePop>(dlsym(instance.handle, "nvtxRangePop"));
        instance.available = (instance.push != nullptr) && (instance.pop != nullptr);
    });
    return instance;
}
#endif

} // namespace

namespace genesis_nvtx {

bool enabled()
{
#if defined(GENESIS_NVTX)
    return envFlagEnabled("GENESIS_CUDA_NVTX", true);
#else
    return false;
#endif
}

void rangePush(const char *name)
{
#if defined(GENESIS_NVTX) && !defined(GENESIS_NVTX_NO_DLOPEN)
    if (!enabled()) {
        return;
    }
    NvtxApi &a = api();
    if (a.available && (name != nullptr)) {
        a.push(name);
    }
#else
    (void)name;
#endif
}

void rangePop()
{
#if defined(GENESIS_NVTX) && !defined(GENESIS_NVTX_NO_DLOPEN)
    if (!enabled()) {
        return;
    }
    NvtxApi &a = api();
    if (a.available) {
        a.pop();
    }
#endif
}

} // namespace genesis_nvtx

GenesisNvtxRange::GenesisNvtxRange(const char *name)
{
#if defined(GENESIS_NVTX)
    active_ = genesis_nvtx::enabled();
    if (active_) {
        genesis_nvtx::rangePush(name);
    }
#else
    (void)name;
#endif
}

GenesisNvtxRange::~GenesisNvtxRange()
{
#if defined(GENESIS_NVTX)
    if (active_) {
        genesis_nvtx::rangePop();
    }
#endif
}
