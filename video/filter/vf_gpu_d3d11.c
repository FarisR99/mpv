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

struct d3d11_offscreen_ctx
{
    struct ra_ctx *ractx;
    struct mpd3d11_ctx *d3d11;
};

extern const struct m_sub_options d3d11_conf;

static void d3d11_ctx_destroy(void *p)
{
    struct offscreen_ctx *ctx = p;
    struct d3d11_offscreen_ctx *d3d11ctx = ctx->priv;
    struct ra_ctx *ractx = d3d11ctx->ractx;
    struct mpd3d11_ctx *d3d11 = d3d11ctx->d3d11;

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
    talloc_set_destructor(ctx, d3d11_ctx_destroy);
    *ctx = (struct offscreen_ctx){
        .log = log};

    struct ra_ctx *ractx = talloc_zero(ctx, struct ra_ctx);
    ractx->log = ctx->log;
    ractx->global = global;

    struct priv *p = ractx->priv = talloc_zero(ractx, struct priv);
    p->opts_cache = m_config_cache_alloc(ractx, ractx->global, &d3d11_conf);
    p->opts = p->opts_cache->opts;

    struct mpd3d11_ctx *d3d = talloc_zero(ctx, struct mpd3d11_ctx);

    d3d->pllog = mppl_log_create(ctx, log);
    if (!d3d->pllog)
        goto error;

    struct pl_d3d11_params pl_d3d_params = {0};
    struct ra_ctx_opts *ctx_opts = mp_get_config_group(NULL, global, &ra_ctx_conf);
    pl_d3d_params.debug = ctx_opts->debug;
    pl_d3d_params.allow_software = ctx_opts->allow_sw;
    talloc_free(ctx_opts);
    mppl_log_set_probing(d3d->pllog, true);
    mp_verbose(ctx->log, "Creating D3D11 device\n");
    d3d->d3d11 = pl_d3d11_create(d3d->pllog, &pl_d3d_params);
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

    return ctx;
error:
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
