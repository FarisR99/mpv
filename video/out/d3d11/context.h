#pragma once

#include <dxgi.h>

#include "video/out/gpu/context.h"
#include "video/out/gpu/d3d11_helpers.h"
#include "common.h"

struct d3d11_opts {
    int feature_level;
    int warp;
    bool flip;
    int sync_interval;
    char *adapter_name;
    int output_format;
    int color_space;
    bool exclusive_fs;
    int output_mode;
};

struct priv {
    struct d3d11_opts *opts;
    struct m_config_cache *opts_cache;

    struct mp_vo_opts *vo_opts;
    struct m_config_cache *vo_opts_cache;

    struct ra_tex *backbuffer;
    ID3D11Device *device;
    IDXGISwapChain *swapchain;
    struct pl_color_space swapchain_csp;

    int64_t perf_freq;
    unsigned last_sync_refresh_count;
    int64_t last_sync_qpc_time;
    int64_t vsync_duration_qpc;
    int64_t last_submit_qpc;

    struct mp_dxgi_factory_ctx dxgi_ctx;
};

// Get the underlying D3D11 swap chain from an RA context. The returned swap chain is
// refcounted and must be released by the caller.
IDXGISwapChain *ra_d3d11_ctx_get_swapchain(struct ra_ctx *ra);

// Returns true if an 8-bit output format is explicitly requested for
// d3d11-output-format for an RA context.
bool ra_d3d11_ctx_prefer_8bit_output_format(struct ra_ctx *ra);
