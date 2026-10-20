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

#include "common/common.h"
#include "filters/f_autoconvert.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "misc/io_utils.h"
#include "misc/path_utils.h"
#include "options/m_config.h"
#include "options/m_option.h"
#include "options/options.h"
#include "options/path.h"
#include "osdep/io.h"
#include "osdep/timer.h"
#include "stream/stream.h"
#include "video/filter/vf_gpu.h"
#include "video/img_format.h"
#include "video/mp_image_pool.h"
#include "video/out/aspect.h"
#include "video/out/gpu/video.h"
#include "video/out/placebo/ra_pl.h"
#include "video/out/placebo/utils.h"
#include "video/out/vo.h"

#include <libplacebo/cache.h>
#include <libplacebo/renderer.h>

extern const struct offscreen_context offscreen_vk;
extern const struct offscreen_context offscreen_egl;
extern const struct offscreen_context offscreen_angle;
extern const struct offscreen_context offscreen_d3d11;

static const struct offscreen_context *contexts[] = {
#if HAVE_VULKAN
    &offscreen_vk,
#endif
#if HAVE_EGL
    &offscreen_egl,
#endif
#if HAVE_EGL_ANGLE_WIN32
    &offscreen_angle,
#endif
#if HAVE_D3D11
    &offscreen_d3d11,
#endif
};

static inline OPT_STRING_VALIDATE_FUNC(offscreen_ctx_validate_api)
{
    struct bstr param = bstr0(*value);
    for (int i = 0; i < MP_ARRAY_SIZE(contexts); i++) {
        if (bstr_equals0(param, contexts[i]->api))
            return 1;
    }
    return M_OPT_INVALID;
}

static int offscreen_ctx_api_help(struct mp_log *log, const struct m_option *opt,
                                  struct bstr name)
{
    mp_info(log, "GPU APIs (offscreen contexts):\n");
    for (int i = 0; i < MP_ARRAY_SIZE(contexts); i++)
        mp_info(log, "    %s\n", contexts[i]->api);
    return M_OPT_EXIT;
}

static struct offscreen_ctx *offscreen_ctx_create(struct mpv_global *global,
                                                  struct mp_log *log,
                                                  const char *api)
{
    for (int i = 0; i < MP_ARRAY_SIZE(contexts); i++) {
        if (api && strcmp(contexts[i]->api, api) != 0)
            continue;
        mp_info(log, "Creating offscreen GPU context '%s'\n", contexts[i]->api);
        return contexts[i]->offscreen_ctx_create(global, log);
    }
    return NULL;
}

static bool offscreen_ctx_set_current(struct offscreen_ctx *ctx, bool enable)
{
    if (ctx->set_context)
        return ctx->set_context(ctx, enable);
    return true;
}

struct gpu_opts {
    int w, h;
    char *api;
    int hw_output;
    int hw_format;
    int sw_format;
};

// Number of software output frames that can be rendered and downloading at
// once. While waiting for the oldest download, the GPU renders the others.
#define DOWNLOAD_DEPTH 3

struct download_slot {
    struct ra_tex *tex; // RGB output
    pl_tex planes[MP_MAX_PLANES]; // NV12 output
    pl_buf buf;
    struct mp_image *img; // waiting for the download to finish, or NULL
};

struct priv {
    struct gpu_opts *opts;
    struct m_config_cache *vo_opts_cache;
    struct mp_vo_opts *vo_opts;

    struct offscreen_ctx *ctx;
    struct gl_video *renderer;
    // Rendered to for software output and as the source of YUV conversion.
    struct ra_tex *target;
    struct mp_pin *in_pin;

    pl_cache shader_cache;
    pl_log cache_log;
    char *cache_path;
    uint64_t cache_signature;

    struct mp_image_params img_params;
    uint64_t next_frame_id;
    int hw_format; // surface format of hardware output, 0 for software output
    int hw_layout; // offscreen_ctx.get_hw_frame layout of hw_format
    bool hw_output_resolved;

    // Asynchronous software output: a ring of render targets downloaded into
    // persistent buffers, read back only once the next frames are submitted.
    bool async_download;
    int sw_format; // format of software output
    struct download_slot slots[DOWNLOAD_DEPTH];
    int first_pending, num_pending;
    struct mp_image_pool *pool;

    // Converts RGB to YUV hardware output.
    pl_log yuv_log;
    pl_renderer yuv_renderer;
    struct pl_color_repr yuv_repr;
    bool restore_subs_in_filter;
    bool prev_subs_in_filter;

    // Per-frame timing, logged on destruction for benchmarking.
    int64_t num_frames;
    int64_t render_ns, render_max_ns;
    int64_t sync_ns, sync_max_ns;
};

static void add_time(int64_t *total, int64_t *max, int64_t ns)
{
    *total += ns;
    *max = MPMAX(*max, ns);
}

static void shader_cache_init(struct mp_filter *f)
{
    struct priv *p = f->priv;
    pl_gpu gpu = ra_pl_get(p->ctx->ra);
    if (!gpu)
        return;

    struct gl_video_opts *opts = mp_get_config_group(f, f->global, &gl_video_conf);
    char *dir = NULL;
    if (opts->shader_cache) {
        dir = opts->shader_cache_dir && opts->shader_cache_dir[0]
            ? mp_get_user_path(f, f->global, opts->shader_cache_dir)
            : mp_find_user_file(f, f->global, "cache", "");
    }
    talloc_free(opts);
    if (!dir || !dir[0]) {
        talloc_free(dir);
        return;
    }

    mp_mkdirp(dir);
    p->cache_path = mp_path_join(p, dir, "vf_gpu_shader.cache");
    talloc_free(dir);
    // Own the log separately: the cache outlives the offscreen GPU context.
    p->cache_log = mppl_log_create(p, f->log);
    p->shader_cache = pl_cache_create(pl_cache_params(
        .log = p->cache_log,
        .max_total_size = 128 << 20,
        .max_object_size = 128 << 20,
    ));
    struct stat st;
    struct bstr data = {0};
    if (stat(p->cache_path, &st) == 0)
        data = stream_read_file(p->cache_path, NULL, f->global, 256 << 20);
    if (data.len) {
        int count = pl_cache_load(p->shader_cache, data.start, data.len);
        if (count >= 0) {
            MP_VERBOSE(f, "Loaded %d libplacebo shader cache objects from '%s'.\n",
                       count, p->cache_path);
        }
    }
    talloc_free(data.start);
    p->cache_signature = pl_cache_signature(p->shader_cache);
    pl_gpu_set_cache(gpu, p->shader_cache);
}

static void shader_cache_uninit(struct mp_filter *f)
{
    struct priv *p = f->priv;
    if (!p->shader_cache)
        return;

    if (pl_cache_signature(p->shader_cache) != p->cache_signature) {
        size_t size = pl_cache_save(p->shader_cache, NULL, 0);
        uint8_t *data = talloc_size(NULL, size);
        pl_cache_save(p->shader_cache, data, size);
        if (mp_save_to_file(p->cache_path, data, size)) {
            MP_VERBOSE(f, "Saved %d libplacebo shader cache objects to '%s'.\n",
                       pl_cache_objects(p->shader_cache), p->cache_path);
        } else {
            MP_VERBOSE(f, "Could not save libplacebo shader cache to '%s'.\n",
                       p->cache_path);
        }
        talloc_free(data);
    }
    pl_cache_destroy(&p->shader_cache);
    pl_log_destroy(&p->cache_log);
}

static void destroy_planes(pl_gpu gpu, pl_tex planes[MP_MAX_PLANES])
{
    for (int n = 0; n < MP_MAX_PLANES; n++)
        pl_tex_destroy(gpu, &planes[n]);
}

// Hand a frame from get_hw_frame back to the backend. No-op if already done.
static void end_hw_frame(struct offscreen_ctx *ctx)
{
    if (ctx->end_hw_frame)
        ctx->end_hw_frame(ctx);
}

static struct ra_tex *create_target(struct ra *ra, int w, int h, bool yuv)
{
    // YUV conversion samples the target; use 16 bits to avoid rounding
    // before the conversion.
    struct ra_tex_params params = {
        .dimensions = 2,
        .downloadable = !yuv,
        .w = w,
        .h = h,
        .d = 1,
        .render_dst = true,
        .render_src = yuv,
        .src_linear = yuv,
    };

    if (yuv)
        params.format = ra_find_unorm_format(ra, 2, 4);
    if (!params.format || !params.format->renderable ||
        !params.format->linear_filter)
        params.format = ra_find_unorm_format(ra, 1, 4);

    if (!params.format || !params.format->renderable)
        return NULL;

    return ra_tex_create(ra, &params);
}

// Downloads are asynchronous if the GPU can download into buffers.
static bool can_download_async(struct ra *ra)
{
    pl_gpu gpu = ra_pl_get(ra);
    return gpu && gpu->limits.buf_transfer;
}

static void clear_pending(struct priv *priv)
{
    for (int n = 0; n < DOWNLOAD_DEPTH; n++)
        TA_FREEP(&priv->slots[n].img);
    priv->first_pending = priv->num_pending = 0;
}

static void destroy_slots(struct priv *priv)
{
    clear_pending(priv);
    for (int n = 0; n < DOWNLOAD_DEPTH; n++) {
        struct download_slot *slot = &priv->slots[n];
        ra_tex_free(priv->ctx->ra, &slot->tex);
        destroy_planes(ra_pl_get(priv->ctx->ra), slot->planes);
        pl_buf_destroy(ra_pl_get(priv->ctx->ra), &slot->buf);
    }
}

static struct download_slot *next_slot(struct priv *priv)
{
    mp_assert(priv->num_pending < DOWNLOAD_DEPTH);
    int n = (priv->first_pending + priv->num_pending) % DOWNLOAD_DEPTH;
    return &priv->slots[n];
}

// Create the slot's textures for the software output format.
static bool init_slot(struct priv *priv, struct download_slot *slot, int w, int h)
{
    struct ra *ra = priv->ctx->ra;
    pl_gpu gpu = ra_pl_get(ra);

    if (priv->sw_format != IMGFMT_NV12) {
        if (!slot->tex)
            slot->tex = create_target(ra, w, h, false);
        return slot->tex;
    }

    for (int n = 0; n < 2 && !slot->planes[n]; n++) {
        pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, n + 1, 8, 8,
                                 PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
        if (!fmt)
            return false;
        slot->planes[n] = pl_tex_create(gpu, pl_tex_params(
            .w = w >> n,
            .h = h >> n,
            .format = fmt,
            .renderable = true,
            .host_readable = true,
        ));
        if (!slot->planes[n])
            return false;
    }
    return true;
}

// Return the textures to download, and the offset of each in the buffer.
static int slot_layout(struct download_slot *slot, pl_tex tex[MP_MAX_PLANES],
                       size_t offset[MP_MAX_PLANES + 1])
{
    int num = 0;
    if (slot->tex) {
        tex[num++] = ra_pl_tex_get(slot->tex);
    } else {
        while (num < MP_MAX_PLANES && slot->planes[num]) {
            tex[num] = slot->planes[num];
            num++;
        }
    }

    offset[0] = 0;
    for (int n = 0; n < num; n++) {
        size_t size = tex[n]->params.w * tex[n]->params.format->texel_size *
                      tex[n]->params.h;
        // Keep each plane suitably aligned for buffer copies.
        offset[n + 1] = MP_ALIGN_UP(offset[n] + size, 4096);
    }
    return num;
}

// Start downloading the slot's textures into its buffer.
static bool start_download(struct ra *ra, struct download_slot *slot)
{
    pl_gpu gpu = ra_pl_get(ra);
    pl_tex tex[MP_MAX_PLANES];
    size_t offset[MP_MAX_PLANES + 1];
    int num = slot_layout(slot, tex, offset);

    if (!slot->buf) {
        slot->buf = pl_buf_create(gpu, pl_buf_params(
            .size = offset[num],
            .memory_type = PL_BUF_MEM_HOST,
            .host_readable = true,
        ));
        if (!slot->buf)
            return false;
    }

    for (int n = 0; n < num; n++) {
        if (!pl_tex_download(gpu, pl_tex_transfer_params(
                .tex = tex[n],
                .buf = slot->buf,
                .buf_offset = offset[n],
                .row_pitch = tex[n]->params.w * tex[n]->params.format->texel_size,
            )))
            return false;
    }

    // Start the rendering and download without waiting for them.
    pl_gpu_flush(gpu);
    return true;
}

// Wait for the slot's download and copy it to the slot's image.
static bool finish_download(struct ra *ra, struct download_slot *slot)
{
    pl_gpu gpu = ra_pl_get(ra);
    struct mp_image *img = slot->img;
    pl_tex tex[MP_MAX_PLANES];
    size_t offset[MP_MAX_PLANES + 1];
    int num = slot_layout(slot, tex, offset);

    while (pl_buf_poll(gpu, slot->buf, UINT64_MAX))
        ;

    for (int n = 0; n < num; n++) {
        size_t row_pitch = tex[n]->params.w * tex[n]->params.format->texel_size;
        int rows = tex[n]->params.h;
        if ((size_t)img->stride[n] == row_pitch) {
            if (!pl_buf_read(gpu, slot->buf, offset[n], img->planes[n],
                             row_pitch * rows))
                return false;
            continue;
        }
        for (int y = 0; y < rows; y++) {
            if (!pl_buf_read(gpu, slot->buf, offset[n] + y * row_pitch,
                             img->planes[n] + y * img->stride[n], row_pitch))
                return false;
        }
    }
    return true;
}

// Return the software output format: NV12 if the GPU converts the output,
// otherwise RGB0. In auto mode, NV12 is used when encoding with an encoder
// that accepts it.
static int choose_sw_format(struct mp_filter *f, int w, int h)
{
    struct priv *priv = f->priv;
    int fmt = priv->opts->sw_format;
    if (fmt == IMGFMT_RGB0)
        return fmt;

    if (!priv->async_download || w % 2 || h % 2) {
        if (fmt)
            MP_WARN(f, "NV12 output requires buffer downloads (such as with "
                    "api=vulkan) and an even size; using RGB.\n");
        return IMGFMT_RGB0;
    }
    if (fmt)
        return fmt;

    struct mp_stream_info *info = mp_filter_find_stream_info(f);
    struct vo *vo = info ? info->dr_vo : NULL;
    if (!vo || !vo->driver->encode)
        return IMGFMT_RGB0;
    uint8_t formats[IMGFMT_END - IMGFMT_START];
    vo_query_formats(vo, formats);
    return formats[IMGFMT_NV12 - IMGFMT_START] ? IMGFMT_NV12 : IMGFMT_RGB0;
}

// Return the hardware output surface format: the first candidate an encoder
// accepts directly, or for hw-output=yes, the first one that can be allocated.
// Return 0 for software output.
static int choose_hw_format(struct mp_filter *f, int w, int h)
{
    struct priv *priv = f->priv;
    if (!priv->opts->hw_output || !priv->ctx->get_hw_frame)
        return 0;

    int candidates[] = {IMGFMT_NV12, IMGFMT_BGRA};
    int num_candidates = MP_ARRAY_SIZE(candidates);
    if (priv->opts->hw_format) {
        candidates[0] = priv->opts->hw_format;
        num_candidates = 1;
    }

    struct mp_stream_info *info = mp_filter_find_stream_info(f);
    struct vo *vo = info ? info->dr_vo : NULL;
    bool encoding = vo && vo->driver->encode;
    pl_gpu gpu = ra_pl_get(priv->ctx->ra);
    int num_layouts = MPMAX(priv->ctx->num_hw_layouts, 1);
    int fallback = 0, fallback_layout = 0;
    for (int n = 0; n < num_candidates; n++) {
        for (int layout = 0; layout < num_layouts; layout++) {
            pl_tex planes[MP_MAX_PLANES];
            struct mp_image *img = priv->ctx->get_hw_frame(priv->ctx,
                                        candidates[n], layout, w, h, planes);
            if (!img)
                continue;
            bool ok = encoding &&
                      vo_control(vo, VOCTRL_QUERY_HW_FRAME, img) == VO_TRUE;
            end_hw_frame(priv->ctx);
            destroy_planes(gpu, planes);
            talloc_free(img);
            if (ok) {
                priv->hw_layout = layout;
                return candidates[n];
            }
            if (!fallback) {
                fallback = candidates[n];
                fallback_layout = layout;
            }
        }
    }

    if (priv->opts->hw_output > 0) {
        if (!fallback)
            MP_WARN(f, "Could not allocate hardware frames; using software output.\n");
        priv->hw_layout = fallback_layout;
        return fallback;
    }
    MP_VERBOSE(f, "Encoder does not accept hardware frames from the renderer.\n");
    return 0;
}

// Convert the rendered RGB target to limited range YUV in the output planes.
static bool render_yuv(struct mp_filter *f, pl_tex planes[MP_MAX_PLANES])
{
    struct priv *priv = f->priv;
    pl_gpu gpu = ra_pl_get(priv->ctx->ra);
    if (!priv->yuv_renderer) {
        priv->yuv_log = mppl_log_create(priv, f->log);
        priv->yuv_renderer = pl_renderer_create(priv->yuv_log, gpu);
        if (!priv->yuv_renderer)
            return false;
    }

    const struct mp_image_params *params =
        gl_video_get_target_params_ptr(priv->renderer);
    struct pl_frame src = {
        .num_planes = 1,
        .planes = {{
            .texture = ra_pl_tex_get(priv->target),
            .components = 3,
            .component_mapping = {PL_CHANNEL_R, PL_CHANNEL_G, PL_CHANNEL_B},
        }},
        .repr = pl_color_repr_rgb,
        .color = params->color,
    };

    priv->yuv_repr = (struct pl_color_repr) {
        .sys = params->color.primaries == PL_COLOR_PRIM_BT_2020
            ? PL_COLOR_SYSTEM_BT_2020_NC : PL_COLOR_SYSTEM_BT_709,
        .levels = PL_COLOR_LEVELS_LIMITED,
        .bits = {.sample_depth = 8, .color_depth = 8},
    };
    struct pl_frame dst = {
        .num_planes = 2,
        .planes = {
            {
                .texture = planes[0],
                .components = 1,
                .component_mapping = {PL_CHANNEL_Y},
            },
            {
                .texture = planes[1],
                .components = 2,
                .component_mapping = {PL_CHANNEL_CB, PL_CHANNEL_CR},
            },
        },
        .repr = priv->yuv_repr,
        .color = params->color,
    };
    pl_frame_set_chroma_location(&dst, PL_CHROMA_LEFT);

    return pl_render_image(priv->yuv_renderer, &src, &dst, pl_render_params(
        .plane_downscaler = &pl_filter_hermite,
        .dither_params = &pl_dither_default_params,
    ));
}

static struct mp_image *gpu_render_frame(struct mp_filter *f, struct mp_image *in)
{
    struct priv *priv = f->priv;
    bool ok = false;
    struct mp_image *res = NULL;
    struct ra *ra = priv->ctx->ra;

    if (priv->opts->w <= 0)
        priv->opts->w = in->w;
    if (priv->opts->h <= 0)
        priv->opts->h = in->h;

    int w = priv->opts->w;
    int h = priv->opts->h;

    struct vo_frame frame = {
        .pts = in->pts,
        .duration = -1,
        .num_vsyncs = 1,
        .current = in,
        .num_frames = 1,
        .frames = {in},
        .frame_id = ++(priv->next_frame_id),
        .can_drop = false,
    };

    bool need_reconfig = m_config_cache_update(priv->vo_opts_cache);

    if (!mp_image_params_static_equal(&priv->img_params, &in->params)) {
        gl_video_config(priv->renderer, &in->params);
        need_reconfig = true;
    }

    if (!mp_image_params_equal(&priv->img_params, &in->params))
        priv->img_params = in->params;

    if (need_reconfig) {
        struct mp_rect src, dst;
        struct mp_osd_res osd;

        struct mp_stream_info *info = mp_filter_find_stream_info(f);
        struct osd_state *osd_state = info ? info->osd : NULL;
        if (osd_state) {
            if (!priv->restore_subs_in_filter) {
                priv->prev_subs_in_filter =
                    osd_get_render_subs_in_filter(osd_state);
                priv->restore_subs_in_filter = true;
            }
            osd_set_render_subs_in_filter(osd_state, true);
            // Assume the osd_state doesn't somehow disappear.
            gl_video_set_osd_source(priv->renderer, osd_state);
        }

        mp_get_src_dst_rects(f->log, priv->vo_opts, VO_CAP_ROTATE90, &in->params,
                             w, h, 1, &src, &dst, &osd);

        gl_video_resize(priv->renderer, &src, &dst, &osd);
    }

    if (!priv->hw_output_resolved) {
        priv->hw_format = choose_hw_format(f, w, h);
        priv->hw_output_resolved = true;
        if (priv->hw_format) {
            MP_VERBOSE(f, "Rendering directly to %s hardware frames.\n",
                       mp_imgfmt_to_name(priv->hw_format));
        } else {
            priv->async_download = can_download_async(ra);
            priv->sw_format = choose_sw_format(f, w, h);
            MP_VERBOSE(f, "Downloading %s frames%s.\n",
                       mp_imgfmt_to_name(priv->sw_format),
                       priv->async_download ? " asynchronously" : "");
        }
    }
    bool yuv = priv->hw_format == IMGFMT_NV12 || priv->sw_format == IMGFMT_NV12;

    pl_tex planes[MP_MAX_PLANES] = {0};
    struct ra_tex hw_target;
    struct ra_tex *fbo = priv->target;
    struct download_slot *slot = NULL;
    if (priv->hw_format) {
        res = priv->ctx->get_hw_frame(priv->ctx, priv->hw_format,
                                      priv->hw_layout, w, h, planes);
        if (!res)
            goto done;
    }

    if (priv->hw_format == IMGFMT_BGRA) {
        // Render directly into the hardware frame.
        if (!mppl_wrap_tex(ra, planes[0], &hw_target))
            goto done;
        fbo = &hw_target;
    } else {
        // Each frame in flight has its own textures to download from, so that
        // rendering the next frame doesn't wait for the previous download.
        // NV12 is converted from the shared target.
        if (priv->async_download) {
            slot = next_slot(priv);
            if (!init_slot(priv, slot, w, h))
                goto done;
        }
        if (slot && slot->tex) {
            fbo = slot->tex;
        } else if (!priv->target) {
            priv->target = create_target(ra, w, h, yuv);
            if (!priv->target)
                goto done;
            fbo = priv->target;
        }
    }

    // (it doesn't have access to the OSD though)
    int flags = RENDER_FRAME_SUBS | RENDER_FRAME_VF_SUBS;
    gl_video_render_frame(priv->renderer, &frame, &(struct ra_fbo){fbo}, flags);

    if (yuv && !render_yuv(f, slot ? slot->planes : planes))
        goto done;

    int64_t sync_start = mp_time_ns();
    if (priv->hw_format) {
        // Submit the rendering without waiting for it. With D3D11 the consumer
        // (such as NVENC) uses the same device, so its reads are ordered after
        // these writes; with Vulkan, end_hw_frame signals the frame's
        // semaphore for it. The AVHWFramesContext retains the texture until
        // the consumer is done.
        end_hw_frame(priv->ctx);
        pl_gpu_flush(ra_pl_get(ra));
        add_time(&priv->sync_ns, &priv->sync_max_ns, mp_time_ns() - sync_start);
        ok = true;
        goto done;
    }

    if (slot) {
        // gpu_process() queues the frame; finish_download() fills it later.
        if (!start_download(ra, slot))
            goto done;
        if (!priv->pool)
            priv->pool = mp_image_pool_new(priv);
        res = mp_image_pool_get(priv->pool, priv->sw_format, w, h);
        ok = !!res;
        goto done;
    }

    res = mp_image_alloc(IMGFMT_RGB0, w, h);
    if (!res)
        goto done;

    struct ra_tex_download_params download_params = {
        .tex = priv->target,
        .dst = res->planes[0],
        .stride = res->stride[0],
    };
    if (!ra->fns->tex_download(ra, &download_params))
        goto done;
    add_time(&priv->sync_ns, &priv->sync_max_ns, mp_time_ns() - sync_start);

    ok = true;
done:
    if (priv->hw_format)
        end_hw_frame(priv->ctx);
    destroy_planes(ra_pl_get(ra), planes);
    if (priv->ctx->debug_marker)
        priv->ctx->debug_marker(priv->ctx, "after rendering");
    if (!ok)
        TA_FREEP(&res);
    return res;
}

// Output the oldest frame waiting for its download.
static void output_pending(struct mp_filter *f)
{
    struct priv *priv = f->priv;
    struct download_slot *slot = &priv->slots[priv->first_pending];
    priv->first_pending = (priv->first_pending + 1) % DOWNLOAD_DEPTH;
    priv->num_pending--;

    int64_t start = mp_time_ns();
    bool ok = offscreen_ctx_set_current(priv->ctx, true);
    ok = ok && finish_download(priv->ctx->ra, slot);
    ok = offscreen_ctx_set_current(priv->ctx, false) && ok;
    add_time(&priv->sync_ns, &priv->sync_max_ns, mp_time_ns() - start);

    struct mp_image *res = slot->img;
    slot->img = NULL;
    if (!ok) {
        MP_ERR(f, "Could not retrieve frame.\n");
        talloc_free(res);
        mp_filter_internal_mark_failed(f);
        return;
    }

    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, res));
}

static void gpu_process(struct mp_filter *f)
{
    struct priv *priv = f->priv;
    bool context_current = false;

    if (priv->num_pending == DOWNLOAD_DEPTH) {
        if (mp_pin_in_needs_data(f->ppins[1]))
            output_pending(f);
        return;
    }

    if (!mp_pin_can_transfer_data(f->ppins[1], priv->in_pin))
        return;

    struct mp_frame frame = mp_pin_out_read(priv->in_pin);

    if (mp_frame_is_signaling(frame)) {
        if (priv->num_pending) {
            // Output the frames rendered before the signal first.
            mp_pin_out_unread(priv->in_pin, frame);
            output_pending(f);
            return;
        }
        mp_pin_in_write(f->ppins[1], frame);
        return;
    }

    if (frame.type != MP_FRAME_VIDEO)
        goto error;

    if (!offscreen_ctx_set_current(priv->ctx, true))
        goto error;
    context_current = true;

    struct mp_image *mpi = frame.data;
    int64_t render_start = mp_time_ns();
    struct mp_image *res = gpu_render_frame(f, mpi);
    if (!res) {
        MP_ERR(f, "Could not render or retrieve frame.\n");
        goto error;
    }
    add_time(&priv->render_ns, &priv->render_max_ns, mp_time_ns() - render_start);
    priv->num_frames++;

    // Preserve frame timing, but describe the rendered pixels rather than the
    // source color space, crop, orientation, or pixel aspect ratio.
    res->pts = mpi->pts;
    res->dts = mpi->dts;
    res->pkt_duration = mpi->pkt_duration;
    res->nominal_fps = mpi->nominal_fps;
    res->pict_type = mpi->pict_type;
    res->fields = mpi->fields;

    const struct mp_image_params *target =
        gl_video_get_target_params_ptr(priv->renderer);
    res->params.color = target->color;
    if (priv->hw_format == IMGFMT_NV12 || priv->sw_format == IMGFMT_NV12) {
        res->params.repr = priv->yuv_repr;
        res->params.chroma_location = PL_CHROMA_LEFT;
    } else {
        res->params.repr.sys = target->repr.sys;
        res->params.repr.levels = target->repr.levels;
    }
    res->params.light = target->light;

    if (!offscreen_ctx_set_current(priv->ctx, false)) {
        talloc_free(res);
        goto error;
    }
    context_current = false;

    talloc_free(mpi);

    if (priv->async_download) {
        // Render the next frames while this one downloads.
        next_slot(priv)->img = res;
        priv->num_pending++;
        mp_filter_internal_mark_progress(f);
        return;
    }

    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, res));
    return;

error:
    mp_frame_unref(&frame);
    mp_filter_internal_mark_failed(f);
    if (context_current)
        offscreen_ctx_set_current(priv->ctx, false);
}

static void gpu_reset(struct mp_filter *f)
{
    struct priv *priv = f->priv;

    if (!offscreen_ctx_set_current(priv->ctx, true)) {
        mp_filter_internal_mark_failed(f);
        return;
    }
    gl_video_reset(priv->renderer);
    clear_pending(priv);
    if (!offscreen_ctx_set_current(priv->ctx, false))
        mp_filter_internal_mark_failed(f);
}

static void gpu_destroy(struct mp_filter *f)
{
    struct priv *priv = f->priv;

    if (priv->num_frames) {
        double n = priv->num_frames;
        MP_VERBOSE(f, "Timing: %"PRId64" frames, render avg %.3f ms max %.3f ms, "
                   "%s avg %.3f ms max %.3f ms.\n", priv->num_frames,
                   MP_TIME_NS_TO_MS(priv->render_ns) / n,
                   MP_TIME_NS_TO_MS(priv->render_max_ns),
                   priv->hw_format ? "flush" : "download",
                   MP_TIME_NS_TO_MS(priv->sync_ns) / n,
                   MP_TIME_NS_TO_MS(priv->sync_max_ns));
    }

    if (priv->ctx) {
        offscreen_ctx_set_current(priv->ctx, true);

        pl_renderer_destroy(&priv->yuv_renderer);
        gl_video_uninit(priv->renderer);
        ra_tex_free(priv->ctx->ra, &priv->target);
        destroy_slots(priv);

        offscreen_ctx_set_current(priv->ctx, false);
    }

    talloc_free(priv->ctx);
    pl_log_destroy(&priv->yuv_log);
    // GPU destruction may add backend pipeline data to the cache.
    shader_cache_uninit(f);

    if (priv->restore_subs_in_filter) {
        struct mp_stream_info *info = mp_filter_find_stream_info(f);
        struct osd_state *osd = info ? info->osd : NULL;
        if (osd)
            osd_set_render_subs_in_filter(osd, priv->prev_subs_in_filter);
    }
}

static const struct mp_filter_info gpu_filter = {
    .name = "gpu",
    .process = gpu_process,
    .reset = gpu_reset,
    .destroy = gpu_destroy,
    .priv_size = sizeof(struct priv),
};

static struct mp_filter *gpu_create(struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &gpu_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *priv = f->priv;
    priv->opts = talloc_steal(priv, options);
    priv->vo_opts_cache = m_config_cache_alloc(f, f->global, &vo_sub_opts);
    priv->vo_opts = priv->vo_opts_cache->opts;

    priv->ctx = offscreen_ctx_create(f->global, f->log, priv->opts->api);
    if (!priv->ctx) {
        MP_FATAL(f, "Could not create offscreen ra context.\n");
        goto error;
    }

    if (priv->opts->hw_output > 0 && !priv->ctx->get_hw_frame) {
        MP_FATAL(f, "Hardware output requires api=d3d11 (with D3D hardware "
                    "acceleration support) or api=vulkan.\n");
        goto error;
    }

    if (priv->opts->hw_output <= 0 && !priv->ctx->ra->fns->tex_download) {
        MP_FATAL(f, "Offscreen ra context does not support image retrieval.\n");
        goto error;
    }

    if (!offscreen_ctx_set_current(priv->ctx, true))
        goto error;

    shader_cache_init(f);
    priv->renderer = gl_video_init(priv->ctx->ra, f->log, f->global);
    mp_assert(priv->renderer); // can't fail (strangely)

    if (!offscreen_ctx_set_current(priv->ctx, false))
        goto error;

    struct mp_autoconvert *conv = mp_autoconvert_create(f);
    if (!conv)
        goto error;

    // Pass supported software frames through unchanged; download hardware
    // frames and convert unsupported formats before uploading to the renderer.
    for (int imgfmt = IMGFMT_START + 1; imgfmt < IMGFMT_END; imgfmt++) {
        struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(imgfmt);
        if (!(desc.flags & MP_IMGFLAG_HWACCEL) &&
            gl_video_check_format(priv->renderer, imgfmt))
            mp_autoconvert_add_imgfmt(conv, imgfmt, 0);
    }
    mp_pin_connect(conv->f->pins[0], f->ppins[0]);
    priv->in_pin = conv->f->pins[1];

    // MP_WARN(f, "This is experimental. Keep in mind:\n");
    // MP_WARN(f, " - OSD rendering is done in software.\n");
    // MP_WARN(f, " - Encoding will convert the RGB output to yuv420p in software.\n");
    // MP_WARN(f, " - Using this with --vo=gpu will filter the video twice!\n");
    // MP_WARN(f, "   (And you can't prevent this; they use the same options.)\n");
    // MP_WARN(f, " - Some features are simply not supported.\n");

    return f;

error:
    talloc_free(f);
    return NULL;
}

#define OPT_BASE_STRUCT struct gpu_opts
const struct mp_user_filter_entry vf_gpu = {
    .desc = {
        .description = "vo_gpu as filter",
        .name = "gpu",
        .priv_size = sizeof(OPT_BASE_STRUCT),
        .priv_defaults = &(const OPT_BASE_STRUCT){
            .hw_output = -1,
        },
        .options = (const struct m_option[]){
            {"w", OPT_INT(w)},
            {"h", OPT_INT(h)},
            {"hw-output", OPT_CHOICE(hw_output,
                {"auto", -1}, {"no", 0}, {"yes", 1})},
            {"hw-format", OPT_CHOICE(hw_format,
                {"auto", 0}, {"nv12", IMGFMT_NV12}, {"bgra", IMGFMT_BGRA})},
            {"sw-format", OPT_CHOICE(sw_format,
                {"auto", 0}, {"nv12", IMGFMT_NV12}, {"rgb", IMGFMT_RGB0})},
            {"api", OPT_STRING_VALIDATE(api, offscreen_ctx_validate_api),
                .help = offscreen_ctx_api_help},
            {0}
        },
    },
    .create = gpu_create,
};
