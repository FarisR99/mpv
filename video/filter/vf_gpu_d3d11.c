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

#include "osdep/windows_utils.h"

#include "options/m_config.h"
#include "video/filter/vf_gpu.h"
#include "video/out/d3d11/context.h"
#include "video/out/d3d11/ra_d3d11.h"
#include "video/out/gpu/d3d11_helpers.h"
#include "video/out/gpu/spirv.h"
#include "video/out/placebo/ra_pl.h"
#include "video/out/placebo/utils.h"
#include "video/out/w32_common.h"

#if HAVE_D3D_HWACCEL
#include <d3d10.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
#include "video/d3d.h"
#include "video/mp_image.h"
#endif

struct d3d11_offscreen_ctx
{
    struct ra_ctx *ractx;
    struct mpd3d11_ctx *d3d11;
#if HAVE_D3D_HWACCEL
    AVBufferRef *hw_pool;
#endif
};

extern const struct m_sub_options d3d11_conf;

#if HAVE_D3D_HWACCEL
static bool init_hw_pool(struct offscreen_ctx *ctx, int w, int h)
{
    struct d3d11_offscreen_ctx *p = ctx->priv;
    if (p->hw_pool) {
        AVHWFramesContext *frames = (void *)p->hw_pool->data;
        if (frames->width == w && frames->height == h)
            return true;
        av_buffer_unref(&p->hw_pool);
    }

    ID3D11Device *device = p->d3d11->d3d11->device;
    // The renderer and encoder can access this device from different threads.
    ID3D10Multithread *multithread = NULL;
    HRESULT hr = ID3D11Device_QueryInterface(device, &IID_ID3D10Multithread,
                                           (void **)&multithread);
    if (FAILED(hr))
        return false;
    ID3D10Multithread_SetMultithreadProtected(multithread, TRUE);
    ID3D10Multithread_Release(multithread);

    AVBufferRef *device_ref = d3d11_wrap_device_ref(device);
    if (!device_ref)
        return false;
    p->hw_pool = av_hwframe_ctx_alloc(device_ref);
    av_buffer_unref(&device_ref);
    if (!p->hw_pool)
        return false;

    AVHWFramesContext *frames = (void *)p->hw_pool->data;
    frames->format = AV_PIX_FMT_D3D11;
    frames->sw_format = AV_PIX_FMT_BGRA;
    frames->width = w;
    frames->height = h;
    // A dynamic pool gives each in-flight encoder frame a separate texture.
    frames->initial_pool_size = 0;
    AVD3D11VAFramesContext *d3d_frames = frames->hwctx;
    d3d_frames->BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (av_hwframe_ctx_init(p->hw_pool) < 0) {
        av_buffer_unref(&p->hw_pool);
        return false;
    }

    return true;
}

static struct mp_image *get_hw_target(struct offscreen_ctx *ctx, int w, int h,
                                     struct ra_tex **target)
{
    struct d3d11_offscreen_ctx *p = ctx->priv;
    pl_gpu gpu = p->d3d11->gpu;
    struct mp_image *img = NULL;
    AVFrame *frame = NULL;
    pl_tex tex = NULL;

    if (!init_hw_pool(ctx, w, h))
        goto error;
    frame = av_frame_alloc();
    if (!frame || av_hwframe_get_buffer(p->hw_pool, frame, 0) < 0)
        goto error;
    frame->sample_aspect_ratio = (AVRational){1, 1};
    img = mp_image_from_av_frame(frame);
    av_frame_free(&frame);
    if (!img)
        goto error;

    tex = pl_d3d11_wrap(gpu, pl_d3d11_wrap_params(
        .tex = (ID3D11Resource *)img->planes[0],
        .array_slice = (intptr_t)img->planes[1],
    ));
    *target = talloc_zero(NULL, struct ra_tex);
    if (!mppl_wrap_tex(ctx->ra, tex, *target) || !(*target)->params.render_dst)
        goto error;
    return img;

error:
    mp_err(ctx->log, "Could not allocate a D3D11 hardware render target.\n");
    pl_tex_destroy(gpu, &tex);
    TA_FREEP(target);
    av_frame_free(&frame);
    talloc_free(img);
    return NULL;
}
#endif

static void d3d11_ctx_destroy(void *p)
{
    struct offscreen_ctx *ctx = p;
    struct d3d11_offscreen_ctx *d3d11ctx = ctx->priv;
    struct ra_ctx *ractx = d3d11ctx->ractx;
    struct mpd3d11_ctx *d3d11 = d3d11ctx->d3d11;

#if HAVE_D3D_HWACCEL
    av_buffer_unref(&d3d11ctx->hw_pool);
#endif

    if (ractx->ra)
    {
        pl_gpu_finish(d3d11->gpu);
        ractx->ra->fns->destroy(ctx->ra);
        ractx->ra = NULL;
        ctx->ra = NULL;
    }

    d3d11->gpu = NULL;
    pl_d3d11_destroy(&d3d11->d3d11);
    pl_log_destroy(&d3d11->pllog);
    talloc_free(d3d11);
    talloc_free(ractx);
}

static struct offscreen_ctx *d3d11_offscreen_ctx_create(struct mpv_global *global,
                                                        struct mp_log *log)
{
    struct offscreen_ctx *ctx = talloc(NULL, struct offscreen_ctx);
    *ctx = (struct offscreen_ctx){
        .log = log};

    struct ra_ctx *ractx = talloc_zero(ctx, struct ra_ctx);
    ractx->log = ctx->log;
    ractx->global = global;

    struct priv *p = ractx->priv = talloc_zero(ractx, struct priv);
    p->opts_cache = m_config_cache_alloc(ractx, ractx->global, &d3d11_conf);
    p->opts = p->opts_cache->opts;

    struct mpd3d11_ctx *d3d = talloc_zero(ctx, struct mpd3d11_ctx);
    IDXGIAdapter1 *adapter = NULL;

    d3d->pllog = mppl_log_create(ctx, log);
    if (!d3d->pllog)
        goto error;

    struct pl_d3d11_params pl_d3d_params = {0};
    struct ra_ctx_opts *ctx_opts = mp_get_config_group(NULL, global, &ra_ctx_conf);
    pl_d3d_params.debug = ctx_opts->debug;
    // Keep the existing gpu-sw policy for auto, but honor explicit WARP choices.
    pl_d3d_params.allow_software = p->opts->warp == 1 ||
                                  (p->opts->warp < 0 && ctx_opts->allow_sw);
    pl_d3d_params.force_software = p->opts->warp == 1;
    pl_d3d_params.max_feature_level = p->opts->feature_level;
    talloc_free(ctx_opts);

    if (!pl_d3d_params.force_software && p->opts->adapter_name &&
        p->opts->adapter_name[0])
    {
        adapter = mp_get_dxgi_adapter(log, bstr0(p->opts->adapter_name), NULL);
        if (!adapter)
            mp_warn(log, "Requested D3D11 adapter was not found; using the default.\n");
        pl_d3d_params.adapter = (IDXGIAdapter *)adapter;
    }
    mppl_log_set_probing(d3d->pllog, true);
    mp_verbose(ctx->log, "Creating D3D11 device\n");
    d3d->d3d11 = pl_d3d11_create(d3d->pllog, &pl_d3d_params);
    SAFE_RELEASE(adapter);
    mppl_log_set_probing(d3d->pllog, false);
    if (!d3d->d3d11)
        goto error;

    p->device = d3d->d3d11->device;
    if (!p->device)
        goto error;

    d3d->gpu = d3d->d3d11->gpu;
    ractx->ra = ra_create_pl(d3d->gpu, ractx->log);
    if (!ractx->ra)
        goto error;

    struct d3d11_offscreen_ctx *d3d11ctx = talloc_zero(ctx, struct d3d11_offscreen_ctx);
    *d3d11ctx = (struct d3d11_offscreen_ctx){
        .ractx = ractx,
        .d3d11 = d3d};

    ctx->ra = ractx->ra;
    ctx->priv = d3d11ctx;
#if HAVE_D3D_HWACCEL
    ctx->get_hw_target = get_hw_target;
#endif
    talloc_set_destructor(ctx, d3d11_ctx_destroy);

    return ctx;
error:
    SAFE_RELEASE(adapter);
    pl_d3d11_destroy(&d3d->d3d11);
    pl_log_destroy(&d3d->pllog);
    talloc_free(d3d);
    talloc_free(ractx);
    talloc_free(ctx);
    return NULL;
}

const struct offscreen_context offscreen_d3d11 = {
    .api = "d3d11",
    .offscreen_ctx_create = d3d11_offscreen_ctx_create};
