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

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_angle.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dwmapi.h>

#include "common/common.h"
#include "osdep/windows_utils.h"
#include "video/filter/vf_gpu.h"
#include "video/out/opengl/angle_dynamic.h"
#include "video/out/opengl/common.h"
#include "video/out/opengl/context.h"
#include "video/out/opengl/egl_helpers.h"
#include "video/out/opengl/ra_gl.h"
#include "video/out/opengl/utils.h"
#include "video/out/gpu/d3d11_helpers.h"

struct gl_offscreen_ctx
{
    GL gl;

    ID3D11Device *d3d11_device;
    ID3D11DeviceContext *d3d11_context;

    EGLConfig egl_config;
    EGLDisplay egl_display;
    EGLDeviceEXT egl_device;
    EGLContext egl_context;
};

static void d3d11_device_destroy(struct offscreen_ctx *ctx)
{
    struct gl_offscreen_ctx *p = ctx->priv;

    PFNEGLRELEASEDEVICEANGLEPROC eglReleaseDeviceANGLE =
        (PFNEGLRELEASEDEVICEANGLEPROC)eglGetProcAddress("eglReleaseDeviceANGLE");

    if (p->egl_display)
        eglTerminate(p->egl_display);
    p->egl_display = EGL_NO_DISPLAY;

    if (p->egl_device && eglReleaseDeviceANGLE)
        eglReleaseDeviceANGLE(p->egl_device);
    p->egl_device = 0;

    SAFE_RELEASE(p->d3d11_device);
}

static bool d3d11_device_create(struct offscreen_ctx *ctx)
{
    struct gl_offscreen_ctx *p = ctx->priv;

    struct d3d11_device_opts device_opts = {
        .allow_warp = -1 != 0,
        .force_warp = -1 == 1,
        .max_feature_level = D3D_FEATURE_LEVEL_11_0,
        .min_feature_level = D3D_FEATURE_LEVEL_9_3,
        .max_frame_latency = 3,
    };
    if (!mp_d3d11_create_present_device(ctx->log, &device_opts, &p->d3d11_device))
        return false;
    ID3D11Device_GetImmediateContext(p->d3d11_device, &p->d3d11_context);

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglGetPlatformDisplayEXT)
    {
        MP_FATAL(ctx, "Missing EGL_EXT_platform_base\n");
        return false;
    }
    PFNEGLCREATEDEVICEANGLEPROC eglCreateDeviceANGLE =
        (PFNEGLCREATEDEVICEANGLEPROC)eglGetProcAddress("eglCreateDeviceANGLE");
    if (!eglCreateDeviceANGLE)
    {
        MP_FATAL(ctx, "Missing EGL_EXT_platform_device\n");
        return false;
    }

    p->egl_device = eglCreateDeviceANGLE(EGL_D3D11_DEVICE_ANGLE,
                                         p->d3d11_device, NULL);
    if (!p->egl_device)
    {
        MP_FATAL(ctx, "Couldn't create EGL device\n");
        return false;
    }

    p->egl_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                              p->egl_device, NULL);
    if (!p->egl_display)
    {
        MP_FATAL(ctx, "Couldn't get EGL display\n");
        return false;
    }

    return true;
}

static void gl_ctx_destroy(void *p)
{
    struct offscreen_ctx *ctx = p;
    struct gl_offscreen_ctx *gl = ctx->priv;

    // Clean up any potentially left-over debug callback
    if (ctx->ra)
        ra_gl_set_debug(ctx->ra, false);
    ra_free(&ctx->ra);

    // Uninit the EGL device implementation that is being used
    d3d11_device_destroy(ctx);

    if (gl->egl_context)
        eglDestroyContext(gl->egl_display, gl->egl_context);
}

static void gl_ctx_set_context(struct offscreen_ctx *ctx, bool enable)
{
    struct gl_offscreen_ctx *gl = ctx->priv;
    EGLContext c = enable ? gl->egl_context : EGL_NO_CONTEXT;

    if (!eglMakeCurrent(gl->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, c))
        MP_ERR(ctx, "Could not make EGL context current.\n");
}

static bool context_init(struct offscreen_ctx *ctx)
{
    struct gl_offscreen_ctx *gl = ctx->priv;

    // This appears to work with Mesa. EGL 1.5 doesn't specify what a "default
    // display" is at all.
    gl->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(gl->egl_display, NULL, NULL))
    {
        MP_ERR(ctx, "Could not initialize EGL.\n");
        return false;
    }

    const char *exts = eglQueryString(gl->egl_display, EGL_EXTENSIONS);
    if (exts)
        MP_DBG(ctx, "EGL extensions: %s\n", exts);
    return true;
}

static struct offscreen_ctx *gl_offscreen_ctx_create(struct mpv_global *global,
                                                     struct mp_log *log)
{
    struct offscreen_ctx *ctx = talloc(NULL, struct offscreen_ctx);
    struct gl_offscreen_ctx *gl = talloc_zero(ctx, struct gl_offscreen_ctx);
    talloc_set_destructor(ctx, gl_ctx_destroy);
    *ctx = (struct offscreen_ctx){
        .log = log,
        .priv = gl,
        .set_context = gl_ctx_set_context,
    };

    if (!angle_load())
    {
        mp_verbose(log, "Failed to load LIBEGL.DLL\n");
        goto error;
    }

    // Create the underlying EGL device implementation
    bool context_ok = d3d11_device_create(ctx);
    if (context_ok)
    {
        MP_VERBOSE(ctx, "Using Direct3D 11\n");
        context_ok = context_init(ctx);
        if (!context_ok)
            d3d11_device_destroy(ctx);
    }
    if (!context_ok)
    {
        MP_ERR(ctx, "Could not initialise offscreen context.\n");
        goto error;
    }

    // Unfortunately, mpegl_create_context() is entangled with ra_ctx.
    // Fortunately, it does not need much, and we can provide a stub.
    struct ra_ctx ractx = {
        .log = ctx->log,
        .global = global,
    };
    if (!mpegl_create_context(&ractx, gl->egl_display, &gl->egl_context,
                              &gl->egl_config))
    {
        MP_FATAL(ctx, "Could not create EGL context!\n");
        goto error;
    }

    if (!eglMakeCurrent(gl->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                        gl->egl_context))
    {
        MP_ERR(ctx, "Could not make EGL context current.\n");
        goto error;
    }

    mpegl_load_functions(&gl->gl, ctx->log);

    ctx->ra = ra_create_gl(&gl->gl, ctx->log);
    if (!ctx->ra)
        goto error;

    gl_ctx_set_context(ctx, false);

    return ctx;

error:
    d3d11_device_destroy(ctx);
    talloc_free(ctx);
    return NULL;
}

const struct offscreen_context offscreen_angle = {
    .api = "angle",
    .offscreen_ctx_create = gl_offscreen_ctx_create};
