/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <stdint.h>

#include <libplacebo/gpu.h>

struct mp_log;

// NVIDIA RTX Video TrueHDR (SDR to HDR) on libplacebo Vulkan textures. The
// textures are shared with CUDA, and TrueHDR runs through NVEncNVSDKNGX.dll,
// which wraps the RTX Video SDK (nvngx_truehdr.dll).
struct vf_gpu_ngx;

struct vf_gpu_ngx_params {
    uint32_t contrast;      // 0-200
    uint32_t saturation;    // 0-200
    uint32_t middle_gray;   // 10-100
    uint32_t max_luminance; // 400-2000 nits
};

// The output is linear light, with 1.0 at this many nits (measured: with
// this scale, peaks land at max_luminance).
#define VF_GPU_NGX_LINEAR_NITS 100.0f

struct vf_gpu_ngx *vf_gpu_ngx_create(struct mp_log *log, pl_gpu gpu,
                                     const struct vf_gpu_ngx_params *params);
void vf_gpu_ngx_destroy(struct vf_gpu_ngx **ngx);

// Return the w x h 8-bit RGBA texture to render the SDR input into.
pl_tex vf_gpu_ngx_get_input(struct vf_gpu_ngx *ngx, int w, int h);

// Run TrueHDR on the input texture. Returns the half float RGBA output, in
// the input's primaries, or NULL on error. The GPU work is queued; Vulkan and
// CUDA wait for each other through semaphores.
pl_tex vf_gpu_ngx_process(struct vf_gpu_ngx *ngx);
