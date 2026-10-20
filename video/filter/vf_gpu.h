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

#include <libplacebo/gpu.h>

#include "common/common.h"
#include "video/mp_image.h"

struct mpv_global;
struct mp_image;
struct ra_tex;

struct offscreen_ctx {
    struct mp_log *log;
    struct ra *ra;
    void *priv;

    bool (*set_context)(struct offscreen_ctx *ctx, bool enable);
    void (*debug_marker)(struct offscreen_ctx *ctx, const char *msg);

    // Allocate a hardware frame with the given surface format (IMGFMT_BGRA or
    // IMGFMT_NV12) and wrap each of its planes as a renderable texture. The
    // caller owns the frame and the textures, and must submit rendering before
    // destroying the textures. Returns NULL if the format is unsupported.
    // layout picks one of num_hw_layouts ways to allocate the frame (Vulkan:
    // 0 is one multi-planar image, 1 is one image per plane, as CUDA needs).
    struct mp_image *(*get_hw_frame)(struct offscreen_ctx *ctx, int imgfmt,
                                     int layout, int w, int h,
                                     pl_tex planes[MP_MAX_PLANES]);
    int num_hw_layouts; // 0 means 1

    // If set, called after rendering into the textures from get_hw_frame and
    // before destroying them, to hand the frame over to its consumer.
    void (*end_hw_frame)(struct offscreen_ctx *ctx);
};

struct offscreen_context {
    const char *api;
    struct offscreen_ctx *(*offscreen_ctx_create)(struct mpv_global *,
                                                  struct mp_log *);
};
