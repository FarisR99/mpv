#pragma once

#include <libplacebo/d3d11.h>

struct mpd3d11_ctx
{
    pl_log pllog;
    pl_d3d11 d3d11;
    pl_gpu gpu; // points to d3d11->gpu for convenience
    pl_swapchain swapchain;
};
